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
from test_fuzzy_controller_live import FuzzyHWClass
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


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline \
                          version of sumo")
    options, args = opt_parser.parse_args()
    return options


# contains TraCI control loop
def run():
    fuzzyLogic = FuzzyHWClass()
    step = 0

    veh1_gap = []
    veh2_gap = []
    veh3_gap = []
    veh4_gap = []

    veh1_gap_error = []
    veh2_gap_error = []
    veh3_gap_error = []
    veh4_gap_error = []

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        # traci.vehicle.setTau("Car01", "1")
        # traci.vehicle.setTau("Car", "1")
        vehPosition = []
        vehSpeed = []
        if 30 < step < 60:
            for ind in traci.vehicle.getIDList():
                vehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                vehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
            veh1Previous_Gap = (vehPosition[0][0] - 5 - vehPosition[1][0]) / vehSpeed[1]
            veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
            veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
            veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
        elif 59 < step < 150:
            vehPosition = []
            vehSpeed = []
            for ind in traci.vehicle.getIDList():
                vehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                vehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))

            veh1 = fuzzyLogic.calc_Inputs(1, vehPosition[0][0], veh1Previous_Gap, vehPosition[1][0], vehSpeed[1])
            veh1Previous_Gap = veh1[0]
            veh1_gap.append(veh1[0])
            veh1Acceleration = veh1[3]
            veh1Speed = vehSpeed[1] + veh1Acceleration
            veh1_gap_error.append(veh1[1])
            traci.vehicle.setSpeed("1", veh1Speed)

            veh2 = fuzzyLogic.calc_Inputs(2, vehPosition[1][0], veh2Previous_Gap, vehPosition[2][0], vehSpeed[2])
            veh2Previous_Gap = veh2[0]
            veh2_gap.append(veh2[0])
            veh2Acceleration = veh2[3]
            veh2Speed = vehSpeed[2] + veh2Acceleration
            veh2_gap_error.append(veh2[1])
            traci.vehicle.setSpeed("2", veh2Speed)

            veh3 = fuzzyLogic.calc_Inputs(3, vehPosition[2][0], veh3Previous_Gap, vehPosition[3][0], vehSpeed[3])
            veh3Previous_Gap = veh3[0]
            veh3_gap.append(veh3[0])
            veh3Acceleration = veh3[3]
            veh3Speed = vehSpeed[3] + veh3Acceleration
            veh3_gap_error.append(veh3[1])
            traci.vehicle.setSpeed("3", veh3Speed)

            veh4 = fuzzyLogic.calc_Inputs(4, vehPosition[3][0], veh4Previous_Gap, vehPosition[4][0], vehSpeed[4])
            veh4Previous_Gap = veh4[0]
            veh4_gap.append(veh4[0])
            veh4Acceleration = veh4[3]
            veh4Speed = vehSpeed[4] + veh4Acceleration
            veh4_gap_error.append(veh4[1])
            traci.vehicle.setSpeed("4", veh4Speed)

        # if step >= 30 and step < 150:
            # veh1_position = traci.vehicle.getPosition("0")
            # print(veh1_position)
        # close lanes so vehicles do not attmept to pass the slow leader:
        # if step == 120:
            # with the current map, this stop happens between \1/2 or \
            # 2/3 was down the road.
            # traci.vehicle.slowDown("0", "0", "9")
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
    return veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error


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
    veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error = run()
    # convert new xml file to csv

    # inductionLoopFileName = f"{subdirectory}\{recnum}_induction.xml"
    # xml2csv.main([ssmFileName])
    # xml2csv.main([tripInfoFileName])
    xml2csv.main([fcdOutInfoFileName])
    # xml2csv.main([inductionLoopFileName])

    fcdOutCSV = os.path.splitext(fcdOutInfoFileName)[0]+'.csv'
    # test = pull_Results(fcdOutCSV)
    # print(test.veh0Position)
    print(fcdOutCSV)
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

    try:
        os.mkdir(f'./{subdirectory}/images/')
    except Exception:
        pass
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(time0, veh0Position, label="Vehicle 0")
    ax.plot(time1, veh1Position, label="Vehicle 1")
    ax.plot(time2, veh2Position, label="Vehicle 2")
    ax.plot(time3, veh3Position, label="Vehicle 3")
    ax.plot(time4, veh4Position, label="Vehicle 4")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Position')
    ax.legend()
    ax.set_title("FIS Vehcile Position vs Time")
    posFile = f'./{subdirectory}/images/vehicle_position.png'
    if os.path.isfile(posFile):
        os.unlink(posFile)
    fig.savefig(f'{posFile}')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(time0, veh0Velocity, label="Vehicle 0")
    ax.plot(time1, veh1Velocity, label="Vehicle 1")
    ax.plot(time2, veh2Velocity, label="Vehicle 2")
    ax.plot(time3, veh3Velocity, label="Vehicle 3")
    ax.plot(time4, veh4Velocity, label="Vehicle 4")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Velocity')
    ax.legend()
    ax.set_title("FIS Vehcile Velocity vs Time")
    velFile = f'./{subdirectory}/images/vehicle_velocity.png'
    if os.path.isfile(velFile):
        os.unlink(velFile)
    fig.savefig(f'{velFile}')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(time0, veh0Acceleration, label="Vehicle 0")
    ax.plot(time1, veh1Acceleration, label="Vehicle 1")
    ax.plot(time2, veh2Acceleration, label="Vehicle 2")
    ax.plot(time3, veh3Acceleration, label="Vehicle 3")
    ax.plot(time4, veh4Acceleration, label="Vehicle 4")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Acceleration')
    ax.legend()
    ax.set_title("FIS Vehcile Acceleration vs Time")
    accelFile = f'./{subdirectory}/images/vehicle_acceleration.png'
    if os.path.isfile(accelFile):
        os.unlink(accelFile)
    fig.savefig(f'{accelFile}')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(range(90), veh1_gap_error, label="Vehicle 1")
    ax.plot(range(90), veh2_gap_error, label="Vehicle 2")
    ax.plot(range(90), veh3_gap_error, label="Vehicle 3")
    ax.plot(range(90), veh4_gap_error, label="Vehicle 4")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Gap Error')
    ax.legend()
    ax.set_title("FIS Vehcile Gap Error vs Time")
    gapErrFile = f'./{subdirectory}/images/vehicle_gap.png'
    if os.path.isfile(gapErrFile):
        os.unlink(gapErrFile)
    fig.savefig(f'{gapErrFile}')
