#!/usr/bin/env python
#cacc platooning method - anomaly 0
#distance detector anomaly

##############################################
import os
import sys
import optparse
import time
import numpy as np
import matplotlib.pyplot as plt
import random

from GA.chromosome import *
from GA.initializationFunctions import *
from GA.selectionFunctions import *
from GA.crossoverFunctions import *
from GA.mutationFunctions import *
from GA.elitismFunctions import *
from GA.continousGeneticAlgorithm import CGA
from GA.chromosome import *
from fuzzy_tools.CustomFIS import HeiTerry_FIS
#necessary to import xml2csv file from a different directory
#source:https://www.codegrepper.com/code-examples/python/import+script+from+another+folder+python
sys.path.append('C:/Program Files (x86)/Eclipse/Sumo/tools/xml')
#import xml2csv

import numpy as np 

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
import traci.constants as tc


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

def get_noise():
    mu, sigma = 0, 0.5 
    # creating a noise with the same dimension as the dataset (2,2) 
    noise = np.random.normal(mu, sigma) 

# look up a new way to generate xml file.
# https://www.codegrepper.com/code-examples/python/python+string+to+xml
def generate_routefile(routeFileName):
    #creating the rout file.
    with open(routeFileName, "w") as route:
        print('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',file=route)
        print('\t<trip id="vehicle_0" depart="0.00" from="gneE40" to="gneE29"/>',file=route)
        print('\t<trip id="vehicle_1" depart="0.00" from="gneE37" to="gneE29"/>',file=route)
        print('</routes>',file=route)




def generate_additionalfile(additionalFileName, inductionLoopFileName):
    #creating the rout file.
    with open(additionalFileName, "w") as additional:
        print('<additional>', file=additional)
        print('\t<additional>', file=additional)  #\t used to indent in a print statement      
        print('\t\t<inductionLoop id="myLoop0" lane="e3_0" pos="10" freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t\t<inductionLoop id="myLoop1" lane="e3_1" pos="10" freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t\t<inductionLoop id="myLoop2" lane="e3_2" pos="10" freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t</additional>', file=additional)
        print('</additional>', file=additional)
# contains TraCI control loop

def run(chrome):
    step = 0
    dist_total = 0
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        #print(step)
        #det_vehs = traci.inductionloop.getLastStepVehicleIDs("det_0")
        #for veh in det_vehs:
        #    print(veh)
        
        if step == 0:
            traci.vehicle.subscribe("vehicle_0", (tc.VAR_SPEED, tc.VAR_POSITION))
            traci.vehicle.subscribe("vehicle_1", (tc.VAR_SPEED, tc.VAR_POSITION))

            # Construct the FIS here or somewhere else in run()
            rule_base = chrome.string[0]
            rule_base2 = chrome.string[1]

            FIS = HeiTerry_FIS()
            FIS.add_input('distance', np.arange(-10,10,1), 5)
            FIS.add_input('speed', np.arange(-10,10,1), 3)
            FIS.add_output('accel', np.arange(0,20,1), 3)
            FIS.add_output('lane_change', np.arange(0,1,0.1), 3)

            v_str = ['distance', 'speed', 'accel', 'lane_change']
            mfs3 = ['0', '1', '2', '3', '4']
            # Find a way to automate finding num rules per input earlier and for num inputs
            rules_all = []
            for wow in range(5):
                for gee in range(3):
                    rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]]],
                                      ['AND'], [[v_str[2], str(rule_base[(wow * 3) + gee])], 
                                                [v_str[3], str(rule_base2[(wow * 3) + gee])]]]) 

            FIS.generate_mamdani_rule(rules_all)


        ego = traci.vehicle.getSubscriptionResults("vehicle_0")
        #print(ego)
        invader = traci.vehicle.getSubscriptionResults("vehicle_1")
        #print(invader)

        v_ego = ego.get(64)
        d_ego = ego.get(66)
        v_inv = invader.get(64)
        d_inv = invader.get(66)

        if d_ego and v_ego and v_inv and d_inv:
            rel_dist = ((d_ego[0]-d_inv[0])**2 + (d_ego[1]-d_inv[1])**2)**0.5
            rel_speed = v_ego-v_inv
        else: 
            rel_dist = 100
            rel_speed = 0
        #print(rel_dist, rel_speed)
        output = FIS.compute2Plus([['distance', rel_dist],['speed', rel_speed]], ['accel', 'lane_change'])
        accel = 0.5
        lane_change = 0.7
        if lane_change > 0.5:
            lane_change_bool = 1
        else:
                lane_change_bool = 0
        
        try: traci.vehicle.setAccel("vehicle_0",str(accel)) #sets accel of vehicle num, value
        except: pass

        # print(traci.lane.getIDList())
        #print(traci.vehicle.couldChangeLane("vehicle_0", "1", state=None))

        print(step)
        try: couldChange = traci.vehicle.couldChangeLane("vehicle_0", "1", state=None)
        except: couldChange = False
        if lane_change_bool and couldChange:
            traci.vehicle.changeLane("vehicle_0", 1, 45) # vehicle, num lanes, time in other lane

        #subResults = traci.vehicle.getAllSubscriptionResults()
        #print(subResults)

        # close lanes so vehicles do not attmept to pass the slow leader:
        #traci.lane.setDisallowed("e2_0", "passenger")
        #traci.lane.setDisallowed("e2_2", "passenger")
        #if step == 10:
        #    traci.vehicle.slowDown("0", "0", "10")
            
        # if step ==110:
        #     traci.domain.Domain.getParameter("", "0", maxSpeed)
        #     traci.vehicle.changeTarget("1", "e9")
        #     traci.vehicle.changeTarget("3", "e9")
        if step > 5 and step < 20:
            dist_total += rel_dist
        step += 1
    
    #coome back to poi add
    #traci.poi.add("test string", 0, 0, ("1", "0", "0", "0"))
    #print(traci.poi.getIDCount())
    
    fitness = step + 0.1*dist_total
    return fitness

# main entry point
if __name__ == "__main__":
    options = get_options()
    
    #run sumo without gui
    #sumoBinary = checkBinary('sumo')
    
    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:#run sumo with gui
        sumoBinary = checkBinary('sumo')
    
    timestr = time.strftime("%Y%m%d")
    
    #create subdirectory or join it
    subdirectory = timestr + "_tripInfo"
    try:
        os.mkdir(subdirectory)
    except Exception:
        pass
    
    #set the file name based on increamenting value
    i = 0
    while os.path.exists(os.path.join(subdirectory,"%s_tripinfo.xml" % format(int(i), '03d'))):
        i += 1
    recnum = format(int(i), '03d')
    
    
    #generate route file
    routeFileName = os.path.join(subdirectory,"{}.rou.xml".format(recnum))
    #inductionLoopFileName = "{}_induction.xml".format(recnum)
    generate_routefile(routeFileName)
    
    #generate additional file
    additionalFileName = os.path.join(subdirectory,"{}.add.xml".format(recnum))
    inductionLoopFileName = "{}_induction.xml".format(recnum)
    generate_additionalfile(additionalFileName, inductionLoopFileName)
    
    
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "roundy.sumocfg", "--route-files", routeFileName,
                             #"--additional-files", additionalFileName,
                             "--collision.mingap-factor", "0",
                             "--tripinfo-output", os.path.join(subdirectory,"{}_tripinfo.xml".format(recnum))])
                             #"--emission-output",os.path.join(subdirectory,"{}_emissions.xml".format(recnum))])])
    myCGA = CGA(NumberOfChrom = 20,
          NumbofGenes = 10,
          maxGen = 10,
          PC = 0.75,
          PM = 0,
          Er = 0.15,
          bounds = None)

    myCGA.initialization(asteriodInitialize4)

    def AsteriodFitness(chrom, bounds):
        fitness = run(chrom)
        return fitness

    myCGA.run(selectionFunction = basicSelection,
        crossoverFunction = AsteriodsCrossoverRand1Point2,
        mutationFunction = asteriodMutation,
        fitnessFunction = AsteriodFitness,
        elitismFunction = ElitismTest)

    best = myCGA.getBestChromosome()

    print("\n\n",best)

    traci.close()
    sys.stdout.flush()

    #convert new xml file to csv
    #tripInfoFileName = os.path.join(subdirectory,"{}_tripinfo.xml".format(recnum))
    #emissionsInfoFileName = os.path.join(subdirectory,"{}_emissions.xml".format(recnum))
    #inductionLoopFileName = os.path.join(subdirectory,"{}_induction.xml".format(recnum))
    # xml2csv.main([tripInfoFileName])
    # xml2csv.main([emissionsInfoFileName])
    # xml2csv.main([inductionLoopFileName])
