#!/usr/bin/env python

import matplotlib.pyplot as plt
import ntpath
import os
import optparse
import pandas as pd
import sys
import time
import traci
from sumolib import checkBinary  # Checks for the binary in environ vars
# import xml.etree.ElementTree as ET
# necessary to import xml2csv file from a different directory
# source:https://www.codegrepper.com/code-examples/python/import+script+from+another+folder+python
sys.path.append('C:/Program Files (x86)/Eclipse/Sumo/tools/xml')
import xml2csv  # noqa: E402
# used for writing xml files (better than examples)
# import xml.etree.ElementTree as ET

# need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def pull_Results(fcdOutCSV):
    df_FCD = pd.read_csv(f'{fcdOutCSV}')
    # print(df_FCD)
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
    for index, row in df_FCD.iterrows():
        # print(row["vehicle_id"], row["vehicle_pos"])
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
    return (veh0Position, veh1Position, veh2Position, veh3Position,
            veh4Position, veh0Velocity, veh1Velocity, veh2Velocity,
            veh3Velocity, veh4Velocity)


# look up a new way to generate xml file.
# https://www.codegrepper.com/code-examples/python/python+string+to+xml
def generate_routefile(routeFileName):
    # creating the route file.
    with open(routeFileName, "w") as route:
        # os.path.join(subdirectory,"{}add.xml".format(recnum))
        print('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" \
                xsi:noNamespaceSchemaLocation=\
                "http://sumo.dlr.de/xsd/routes_file.xsd">', file=route)
        print("""\t<vType vClass = "passenger" id="Car" length="5.0"
              maxSpeed="33.528" carFollowModel = "CACC" />""", file=route)
        # speedControlGainCACC = "-0.4"
        # gapClosingControlGainGap = "0.005" gapClosingControlGainGapDot \
        # = "0.05"
        # gapControlGainGap = "0.45" gapControlGainGapDot = "0.0125"
        # collisionAvoidanceGainGap = "0.45" collisionAvoidanceGainGapDot \
        # = "0.05"/>""", file=route)  #\t used to indent in a print statement
        print("""\t<vType vClass = "passenger" id="Car01" length="5.0"
              maxSpeed="33.528" />""", file=route)
        print('\t\t<route id="route01" edges="e0 e1 e2"/>', file=route)
        print('\t\t<vehicle id="0" type="Car" route="route01" depart="0" \
            color="1,0,1"/>', file=route)
        print('\t\t<vehicle id="1" type="Car" route="route01" depart="0" \
            color="0,1,1"/>', file=route)
        print('\t\t<vehicle id="2" type="Car" route="route01" depart="0" \
            color="0,1,1"/>', file=route)
        print('\t\t<vehicle id="3" type="Car" route="route01" depart="0" \
            color="0,1,1"/>', file=route)
        print('\t\t<vehicle id="4" type="Car" route="route01" depart="0" \
            color="0,1,1"/>', file=route)
        print('</routes>', file=route)


def generate_additionalfile(additionalFileName, inductionLoopFileName):
    # creating the rout file.
    with open(additionalFileName, "w") as additional:
        # os.path.join(subdirectory,"{}add.xml".format(recnum))
        print('<additional>', file=additional)
        print('\t<additional>', file=additional)
        # \t used to indent in a print statement
        print('\t\t<inductionLoop id="myLoop0" lane="e2_0" pos="10" \
            freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t</additional>', file=additional)
        print('</additional>', file=additional)


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline \
                          version of sumo")
    options, args = opt_parser.parse_args()
    return options


# contains TraCI control loop
def run():
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step >= 30 and step < 150:
            veh1_position = traci.vehicle.getPosition("0")
            print(veh1_position)
        # close lanes so vehicles do not attmept to pass the slow leader:
        if step == 60:
            # with the current map, this stop happens between \1/2 or \
            # 2/3 was down the road.
            traci.vehicle.slowDown("0", "0", "9")
            # a time of 8 seconds with a decel of 9m/s causes the leading \
            # vehicle to travel for ~68meters before stoping
            # DEFAULT_THRESHOLD_TTC is 3 seconds according to: \
            # https://github.com/eclipse/sumo/blob/main/src/microsim/devices/MSDevice_SSM.cpp
        # if step == 120:
        # traci.vehicle.setAccel("0", "1")
        step += 1
    # coome back to poi add
    # traci.poi.add("test string", 0, 0, ("1", "0", "0", "0"))
    # print(traci.poi.getIDCount())
    # print(traci.vehicle.wantsAndCouldChangeLane("1", "2", state=None))
    traci.close()
    sys.stdout.flush()


# main entry point
if __name__ == "__main__":
    options = get_options()
    # run sumo without gui
    # sumoBinary = checkBinary('sumo')
    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        # run sumo with gui
        sumoBinary = checkBinary('sumo-gui')
    fileName = ntpath.basename(__file__)
    fileName_No_Suffix = os.path.splitext(fileName)[0]
    timestr = time.strftime("%Y%m%d")

    # create subdirectory or join it
    subdirectory = f"./results/{timestr}_{fileName_No_Suffix}_tripInfo"
    try:
        os.mkdir(subdirectory)
    except Exception:
        pass

    # set the file name based on increamenting value
    i = 0
    while os.path.exists(os.path.join(subdirectory,
                         "%s_tripinfo.xml" % format(int(i), '03d'))):
        i += 1
    recnum = format(int(i), '03d')
    # another way to seperate new log files:
    # https://sumo.dlr.de/docs/Simulation/Output/index.html#separating_outputs_of_repeated_runs

    # generate route file
    routeFileName = f"{fileName_No_Suffix}.rou.xml"
    # inductionLoopFileName = "{}_induction.xml".format(recnum)
    # generate_routefile(routeFileName)

    # generate additional file
    additionalFileName = f"{fileName_No_Suffix}.add.xml"
    inductionLoopFileName = f"./results/{recnum}_induction.xml"
    # generate_additionalfile(additionalFileName, inductionLoopFileName)

    ssmFileName = rf"{subdirectory}\{recnum}_ssm.xml"
    tripInfoFileName = rf"{subdirectory}\{recnum}_tripinfo.xml"
    fcdOutInfoFileName = rf"{subdirectory}\{recnum}_fcdout.xml"
    amitranInforFileName = rf"{subdirectory}\{recnum}_amitran.xml"
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "cacc_Single_Lane.sumocfg",
                "--route-files", routeFileName,
                 "--additional-files", additionalFileName,
                 # "--collision.mingap-factor", "0",
                 "--device.ssm.probability", "1",
                 "--device.ssm.file", ssmFileName,
                 "--tripinfo-output", tripInfoFileName,
                 "--fcd-output", fcdOutInfoFileName,
                 "--fcd-output.acceleration"])
    run()

    # convert new xml file to csv

    # inductionLoopFileName = f"{subdirectory}\{recnum}_induction.xml"
    # xml2csv.main([ssmFileName])
    # xml2csv.main([tripInfoFileName])
    xml2csv.main([fcdOutInfoFileName])
    # xml2csv.main([inductionLoopFileName])

    fcdOutCSV = os.path.splitext(fcdOutInfoFileName)[0]+'.csv'
    # test = pull_Results(fcdOutCSV)
    # print(test.veh0Position)

    df_FCD = pd.read_csv(f'{fcdOutCSV}')
    time0 = []
    time1 = []
    time2 = []
    time3 = []
    time4 = []
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
    veh0Acceleration = []
    veh1Acceleration = []
    veh2Acceleration = []
    veh3Acceleration = []
    veh4Acceleration = []
    for index, row in df_FCD.iterrows():
        # print(row["vehicle_id"], row["vehicle_pos"])
        if row["vehicle_id"] == 0:
            time0.append(row["timestep_time"])
            veh0Position.append(row["vehicle_x"])
            veh0Velocity.append(row["vehicle_speed"])
            veh0Acceleration.append(row["vehicle_acceleration"])
        elif row["vehicle_id"] == 1:
            time1.append(row["timestep_time"])
            veh1Position.append(row["vehicle_x"])
            veh1Velocity.append(row["vehicle_speed"])
            veh1Acceleration.append(row["vehicle_acceleration"])
        elif row["vehicle_id"] == 2:
            time2.append(row["timestep_time"])
            veh2Position.append(row["vehicle_x"])
            veh2Velocity.append(row["vehicle_speed"])
            veh2Acceleration.append(row["vehicle_acceleration"])
        elif row["vehicle_id"] == 3:
            time3.append(row["timestep_time"])
            veh3Position.append(row["vehicle_x"])
            veh3Velocity.append(row["vehicle_speed"])
            veh3Acceleration.append(row["vehicle_acceleration"])
        elif row["vehicle_id"] == 4:
            time4.append(row["timestep_time"])
            veh4Position.append(row["vehicle_x"])
            veh4Velocity.append(row["vehicle_speed"])
            veh4Acceleration.append(row["vehicle_acceleration"])

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(time0, veh0Position, label="Vehicle 0")
    ax.plot(time1, veh1Position, label="Vehicle 1")
    ax.plot(time2, veh2Position, label="Vehicle 2")
    ax.plot(time3, veh3Position, label="Vehicle 3")
    ax.plot(time4, veh4Position, label="Vehicle 4")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Position')
    ax.legend()
    ax.set_title("CACC Vehcile Position vs Time")
    fig.savefig('./images/vehicle_position.png')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(time0, veh0Velocity, label="Vehicle 0")
    ax.plot(time1, veh1Velocity, label="Vehicle 1")
    ax.plot(time2, veh2Velocity, label="Vehicle 2")
    ax.plot(time3, veh3Velocity, label="Vehicle 3")
    ax.plot(time4, veh4Velocity, label="Vehicle 4")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Velocity')
    ax.legend()
    ax.set_title("CACC Vehcile Velocity vs Time")
    fig.savefig('./images/vehicle_velocity.png')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(time0, veh0Acceleration, label="Vehicle 0")
    ax.plot(time1, veh1Acceleration, label="Vehicle 1")
    ax.plot(time2, veh2Acceleration, label="Vehicle 2")
    ax.plot(time3, veh3Acceleration, label="Vehicle 3")
    ax.plot(time4, veh4Acceleration, label="Vehicle 4")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Acceleration')
    ax.legend()
    ax.set_title("CACC Vehcile Acceleration vs Time")
    fig.savefig('./images/vehicle_acceleration.png')
