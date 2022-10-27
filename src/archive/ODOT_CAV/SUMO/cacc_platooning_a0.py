#!/usr/bin/env python
#cacc platooning method - anomaly 0
#distance detector anomaly

##############################################
import os
import sys
import optparse
import time

#necessary to import xml2csv file from a different directory
#source:https://www.codegrepper.com/code-examples/python/import+script+from+another+folder+python
sys.path.append('C:/Program Files (x86)/Eclipse/Sumo/tools/xml')
import xml2csv

import numpy as np 


#used for writing xml files (better than examples)
#import xml.etree.ElementTree as ET

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary  # Checks for the binary in environ vars
import traci


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
    with open(routeFileName, "w") as route:#os.path.join(subdirectory,"{}add.xml".format(recnum))
#         print("""<additional>
#         <additional>
#             <inductionLoop id="myLoop0" lane="e3_0" pos="10" freq="60" file="inductionLoop.xml" />
#             <inductionLoop id="myLoop1" lane="e3_1" pos="10" freq="60" file="indutionLoop1.xml" />
#             <inductionLoop id="myLoop2" lane="e3_2" pos="10" freq="60" file="inductionLoop2.xml" />
#         </additional>
# </additional>""", file=additional)
        print('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">', file=route)
        print("""\t<vType vClass = "passenger" accel="3.0" decel="2.5" id="Car" length="5.0"
              maxSpeed="10" sigma="0.5" carFollowModel = "CACC" />""", file=route)  #\t used to indent in a print statement      
        print("""\t<vType vClass = "passenger" accel="3.0" decel="2.5" id="Car01" length="5.0" 
              maxSpeed="10" sigma="0.5" carFollowModel = "CACC" speedControlGainCACC = "-0.4" 
              gapClosingControlGainGap = "0.005" gapClosingControlGainGapDot = "0.05"
              gapControlGainGap = "0.45" gapControlGainGapDot = "0.0125" 
              collisionAvoidanceGainGap = "0.45" collisionAvoidanceGainGapDot = "0.05" />""" , file=route)
        print('\t\t<route id="route01" edges="e1 e2 e3 e4"/>', file=route)
        print('\t\t<vehicle id="0" type="Car" route="route01" depart="0" color="1,0,1"/>', file=route)
        print('\t\t<vehicle id="1" type="Car" route="route01" depart="0" color="0,1,1"/>', file=route)
        print('\t\t<vehicle id="2" type="Car" route="route01" depart="0" color="0,1,1"/>', file=route)
        print('\t\t<vehicle id="3" type="Car" route="route01" depart="0" color="0,1,1"/>', file=route)
        print('\t\t<vehicle id="4" type="Car" route="route01" depart="0" color="0,1,1"/>', file=route)
        print('</routes>', file=route)

def generate_additionalfile(additionalFileName, inductionLoopFileName):
    #creating the rout file.
    with open(additionalFileName, "w") as additional:#os.path.join(subdirectory,"{}add.xml".format(recnum))
#         print("""<additional>
#         <additional>
#             <inductionLoop id="myLoop0" lane="e3_0" pos="10" freq="60" file="inductionLoop.xml" />
#             <inductionLoop id="myLoop1" lane="e3_1" pos="10" freq="60" file="indutionLoop1.xml" />
#             <inductionLoop id="myLoop2" lane="e3_2" pos="10" freq="60" file="inductionLoop2.xml" />
#         </additional>
# </additional>""", file=additional)
        print('<additional>', file=additional)
        print('\t<additional>', file=additional)  #\t used to indent in a print statement      
        print('\t\t<inductionLoop id="myLoop0" lane="e3_0" pos="10" freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t\t<inductionLoop id="myLoop1" lane="e3_1" pos="10" freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t\t<inductionLoop id="myLoop2" lane="e3_2" pos="10" freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t</additional>', file=additional)
        print('</additional>', file=additional)
# contains TraCI control loop
def run():
    step = 0
    #traci.vehicle.setAccel("0","1")
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        #print(step)
        #det_vehs = traci.inductionloop.getLastStepVehicleIDs("det_0")
        #for veh in det_vehs:
        #    print(veh)
        #    traci.vehicle.changeLane(veh, 2, 25)
        
        #subResults = traci.vehicle.getAllSubscriptionResults()
        #print(subResults)

        # close lanes so vehicles do not attmept to pass the slow leader:
        traci.lane.setDisallowed("e2_0", "passenger")
        traci.lane.setDisallowed("e2_2", "passenger")
        if step == 10:
            traci.vehicle.slowDown("0", "0", "10")
            
        # if step ==110:
        #     traci.domain.Domain.getParameter("", "0", maxSpeed)
        #     traci.vehicle.changeTarget("1", "e9")
        #     traci.vehicle.changeTarget("3", "e9")

        step += 1
    
    #coome back to poi add
    #traci.poi.add("test string", 0, 0, ("1", "0", "0", "0"))
    #print(traci.poi.getIDCount())
    #print(traci.vehicle.wantsAndCouldChangeLane("1", "2", state=None))
    traci.close()
    sys.stdout.flush()


# main entry point
if __name__ == "__main__":
    options = get_options()
    
    #run sumo without gui
    #sumoBinary = checkBinary('sumo')
    
    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:#run sumo with gui
        sumoBinary = checkBinary('sumo-gui')
    
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
    traci.start([sumoBinary, "-c", "cacc_platooning_a0.sumocfg", "--route-files", routeFileName,
                             "--additional-files", additionalFileName,
                             "--collision.mingap-factor", "0",
                             "--tripinfo-output", os.path.join(subdirectory,"{}_tripinfo.xml".format(recnum)),
                             "--emission-output",os.path.join(subdirectory,"{}_emissions.xml".format(recnum))])
    run()

    #convert new xml file to csv
    tripInfoFileName = os.path.join(subdirectory,"{}_tripinfo.xml".format(recnum))
    emissionsInfoFileName = os.path.join(subdirectory,"{}_emissions.xml".format(recnum))
    inductionLoopFileName = os.path.join(subdirectory,"{}_induction.xml".format(recnum))
    xml2csv.main([tripInfoFileName])
    xml2csv.main([emissionsInfoFileName])
    xml2csv.main([inductionLoopFileName])
