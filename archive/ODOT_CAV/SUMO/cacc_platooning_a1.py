#!/usr/bin/env python
#cacc platooning method - anomaly 0
#distance detector anomaly

##############################################
import csv
import ntpath
import os
import optparse
import pandas as pd
import sys
import time
import xml.etree.ElementTree as ET

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

def pull_Results(fcdOutCSV):
    df = pd.read_csv (f'{fcdOutCSV}')
    #print(df)
    veh0Position = []
    veh1Position = []
    veh2Position = []
    veh3Position = []
    veh4Position = []
    veh0Velocity = []
    veh1Velocity = []
    veh2Velocity = []
    veh3Velocity = []
    veh4Velocity = []
    for index, row in df.iterrows():
        #print(row["vehicle_id"], row["vehicle_pos"])
        if row["vehicle_id"] == 0:
            veh0Position.append(row["vehicle_pos"])
            veh0Velocity.append(row["vehicle_speed"])
        elif row["vehicle_id"] == 1:
            veh1Position.append(row["vehicle_pos"])
            veh1Velocity.append(row["vehicle_speed"])
        elif row["vehicle_id"] == 2:
            veh2Position.append(row["vehicle_pos"])
            veh2Velocity.append(row["vehicle_speed"])
        elif row["vehicle_id"] == 3:
            veh3Position.append(row["vehicle_pos"])
            veh3Velocity.append(row["vehicle_speed"])
        elif row["vehicle_id"] == 4:
            veh4Position.append(row["vehicle_pos"])
            veh4Velocity.append(row["vehicle_speed"])
    return (veh0Position, veh1Position, veh2Position, veh3Position, veh4Position, veh0Velocity, veh1Velocity, veh2Velocity, veh3Velocity, veh4Velocity)

# look up a new way to generate xml file.
# https://www.codegrepper.com/code-examples/python/python+string+to+xml
def generate_routefile(routeFileName):
    #creating the route file.
    with open(routeFileName, "w") as route:#os.path.join(subdirectory,"{}add.xml".format(recnum))
        print('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">', file=route)
        print("""\t<vType vClass = "passenger" id="Car" length="5.0"
              maxSpeed="33.528" carFollowModel = "CACC" />""", file=route)#speedControlGainCACC = "-0.4" 
              #gapClosingControlGainGap = "0.005" gapClosingControlGainGapDot = "0.05"
              #gapControlGainGap = "0.45" gapControlGainGapDot = "0.0125" 
              #collisionAvoidanceGainGap = "0.45" collisionAvoidanceGainGapDot = "0.05"/>""", file=route)  #\t used to indent in a print statement      
        print("""\t<vType vClass = "passenger" id="Car01" length="5.0" 
              maxSpeed="33.528" />""" , file=route)
        print('\t\t<route id="route01" edges="e0 e1 e2"/>', file=route)
        print('\t\t<vehicle id="0" type="Car01" route="route01" depart="0" color="1,0,1"/>', file=route)
        print('\t\t<vehicle id="1" type="Car" route="route01" depart="0" color="0,1,1"/>', file=route)
        print('\t\t<vehicle id="2" type="Car" route="route01" depart="0" color="0,1,1"/>', file=route)
        print('\t\t<vehicle id="3" type="Car" route="route01" depart="0" color="0,1,1"/>', file=route)
        print('\t\t<vehicle id="4" type="Car" route="route01" depart="0" color="0,1,1"/>', file=route)
        print('</routes>', file=route)

def generate_additionalfile(additionalFileName, inductionLoopFileName):
    #creating the rout file.
    with open(additionalFileName, "w") as additional:#os.path.join(subdirectory,"{}add.xml".format(recnum))
        print('<additional>', file=additional)
        print('\t<additional>', file=additional)  #\t used to indent in a print statement      
        print('\t\t<inductionLoop id="myLoop0" lane="e2_0" pos="10" freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t</additional>', file=additional)
        print('</additional>', file=additional)

# contains TraCI control loop
def run():
    step = 0
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        #print(step)

        # close lanes so vehicles do not attmept to pass the slow leader:
        if step == 60: # with the current map, this stop happens between 1/2 or 2/3 was down the road.
            traci.vehicle.slowDown("0", "0", "9") #a time of 8 seconds with a decel of 9m/s causes the leading vehicle to travel for ~68meters before stoping
            #DEFAULT_THRESHOLD_TTC is 3 seconds according to: https://github.com/eclipse/sumo/blob/main/src/microsim/devices/MSDevice_SSM.cpp
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
    fileName = ntpath.basename(__file__)
    timestr = time.strftime("%Y%m%d")
    
    #create subdirectory or join it
    subdirectory = f"{timestr}_{fileName}_tripInfo"
    try:
        os.mkdir(subdirectory)
    except Exception:
        pass
    
    #set the file name based on increamenting value
    i = 0
    while os.path.exists(os.path.join(subdirectory,"%s_tripinfo.xml" % format(int(i), '03d'))):
        i += 1
    recnum = format(int(i), '03d')
    #another way to seperate new log files: https://sumo.dlr.de/docs/Simulation/Output/index.html#separating_outputs_of_repeated_runs
    
    #generate route file
    routeFileName = os.path.join(subdirectory,"{}.rou.xml".format(recnum))
    #inductionLoopFileName = "{}_induction.xml".format(recnum)
    generate_routefile(routeFileName)
    
    #generate additional file
    additionalFileName = os.path.join(subdirectory,"{}.add.xml".format(recnum))
    inductionLoopFileName = "{}_induction.xml".format(recnum)
    generate_additionalfile(additionalFileName, inductionLoopFileName)
    
    
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "cacc_platooning_a1.sumocfg", "--route-files", routeFileName,
                             "--additional-files", additionalFileName,
                             #"--collision.mingap-factor", "0",
                             "--device.ssm.probability", "1",
                             "--device.ssm.file", os.path.join(subdirectory,"{}_ssm.xml".format(recnum)),
                             "--tripinfo-output", os.path.join(subdirectory,"{}_tripinfo.xml".format(recnum)),
                             "--full-output", os.path.join(subdirectory,"{}_fullout.xml".format(recnum)),
                             "--fcd-output", os.path.join(subdirectory,"{}_fcdout.xml".format(recnum)),
                             "--emission-output",os.path.join(subdirectory,"{}_emissions.xml".format(recnum))])
    run()

    #convert new xml file to csv
    ssmFileName = os.path.join(subdirectory,"{}_ssm.xml".format(recnum))
    tripInfoFileName = os.path.join(subdirectory,"{}_tripinfo.xml".format(recnum))
    fullTripInfoFileName = os.path.join(subdirectory,"{}_fullout.xml".format(recnum))
    fcdOutInfoFileName = f"{subdirectory}\{recnum}_fcdout.xml"
    emissionsInfoFileName = os.path.join(subdirectory,"{}_emissions.xml".format(recnum))
    inductionLoopFileName = os.path.join(subdirectory,"{}_induction.xml".format(recnum))
    xml2csv.main([ssmFileName])
    xml2csv.main([tripInfoFileName])
    xml2csv.main([fullTripInfoFileName])
    xml2csv.main([fcdOutInfoFileName])
    xml2csv.main([emissionsInfoFileName])
    xml2csv.main([inductionLoopFileName])

    fcdOutCSV = os.path.splitext(fcdOutInfoFileName)[0]+'.csv'

    test = pull_Results(fcdOutCSV)
    print(test.veh0Position)


    # tree = ET.parse(routeFileName)
    # root = tree.getroot()
    # time_list = []
    # vehicle_list = []
    # speed_list = []
    # for timestep in sumolib.xml.parse(fcdOutInfoFileName, "timestep"):
    #     time_list.append(timestep.time)
    # for vehicle in sumolib.xml.parse(fcdOutInfoFileName, "vehicle"):
    #     speed_list.append(vehicle.speed)

    # #print(len(speed_list))
    # with open(f'{fcdOutCSV}', newline='') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:

    #         #print(row['timestep_time'], row['vehicle_pos'])
    #         vehicle_pos = row['vehicle_pos']