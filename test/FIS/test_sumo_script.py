#!/usr/bin/env python

import matplotlib.pyplot as plt
import ntpath
import numpy as np
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


# UPDATE how this args/options parser is structured. set up similar to graphvi args were parsed
def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline \
                          version of sumo")
    opt_parser.add_option("--krauss", action="store_true",
                          default=False, help="run the simulation using the human driver model")
    opt_parser.add_option("--highway_1", action="store_true",
                          default=False, help="run the simulation using the second highway configuration")
    opt_parser.add_option("--slow_down_midway", action="store_true",
                          default=False, help="slows the leading human model vehicle halfway during the sim")
    options, args = opt_parser.parse_args()
    return options


# contains TraCI control loop
def run(fis_start_time, end_time):
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
        if options.krauss:
            if 30 < step < end_time:
                for ind in traci.vehicle.getIDList():
                    vehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                    vehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                veh1Previous_Gap = (vehPosition[0][0] - 5 - vehPosition[1][0]) / vehSpeed[1]
                veh1_gap_error.append(veh1Previous_Gap-1)
                veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                veh2_gap_error.append(veh2Previous_Gap-1)
                veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                veh3_gap_error.append(veh3Previous_Gap-1)
                veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                veh4_gap_error.append(veh4Previous_Gap-1)
                if options.slow_down_midway:
                    if 525 < step < 615:
                        traci.vehicle.slowDown("0", 20.1168, 90)
                        # traci.vehicle.setSpeed("0", 20.1168)
                    elif 614 < step < 675:
                        traci.vehicle.slowDown("0", 31.292, 60)
                    else:
                        pass
                else:
                    pass
            else:
                veh1_gap_error.append(np.nan)
                veh2_gap_error.append(np.nan)
                veh3_gap_error.append(np.nan)
                veh4_gap_error.append(np.nan)
        else:
            if 30 < step < fis_start_time + 1:
                for ind in traci.vehicle.getIDList():
                    vehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                    vehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                veh1Previous_Gap = (vehPosition[0][0] - 5 - vehPosition[1][0]) / vehSpeed[1]
                veh1_gap_error.append(veh1Previous_Gap-1)
                veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                veh2_gap_error.append(veh2Previous_Gap-1)
                veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                veh3_gap_error.append(veh3Previous_Gap-1)
                veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                veh4_gap_error.append(veh4Previous_Gap-1)

            elif fis_start_time < step < end_time:
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

                if options.slow_down_midway:
                    if 525 < step < 615:
                        traci.vehicle.slowDown("0", 20.1168, 90)
                        # traci.vehicle.setSpeed("0", 20.1168)
                    elif 614 < step < 675:
                        traci.vehicle.slowDown("0", 31.292, 60)
                    else:
                        pass
                else:
                    pass

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

    fileName_No_Suffix = "highway_1"
    fis_start_time = 300
    end_time = 900

    timestr = time.strftime("%Y%m%d")

    # create *_subdirectory or join it
    try:
        os.mkdir("./results/spreadsheet")
    except Exception:
        pass

    try:
        os.mkdir("./results/images")
    except Exception:
        pass

    spreadsheet_subdirectory = f"./results/spreadsheet/{timestr}_{fileName_No_Suffix}_tripInfo"
    images_subdirectory = f"./results/images/{timestr}_{fileName_No_Suffix}_tripInfo"

    try:
        os.mkdir(spreadsheet_subdirectory)
    except Exception:
        pass

    try:
        os.mkdir(images_subdirectory)
    except Exception:
        pass

    # set the file name based on increamenting value
    i = 0
    while os.path.exists(os.path.join(spreadsheet_subdirectory, "%s_fcdout.xml" % format(int(i), '03d'))):
        i += 1
    recnum = format(int(i), '03d')
    # another way to seperate new log files:
    # https://sumo.dlr.de/docs/Simulation/Output/index.html#separating_outputs_of_repeated_runs

    # generate route file
    routeFileName = f"{fileName_No_Suffix}.rou.xml"
    # inductionLoopFileName = "{}_induction.xml".format(recnum)

    # generate additional file
    additionalFileName = f"{fileName_No_Suffix}.add.xml"
    # inductionLoopFileName = f"./results/{recnum}_induction.xml"
    # generate_additionalfile(additionalFileName, inductionLoopFileName)

    ssmFileName = rf"{spreadsheet_subdirectory}\{recnum}_ssm.xml"
    fullOutFileName = rf"{spreadsheet_subdirectory}\{recnum}_fullout.xml"
    fcdOutInfoFileName = rf"{spreadsheet_subdirectory}\{recnum}_fcdout.xml"
    amitranInforFileName = rf"{spreadsheet_subdirectory}\{recnum}_amitran.xml"
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", f"{fileName_No_Suffix}.sumocfg",
                "--route-files", routeFileName,
                 "--additional-files", additionalFileName,
                 "--device.ssm.probability", "1",
                 "--device.ssm.file", ssmFileName,
                 "--full-output", fullOutFileName,
                 "--fcd-output", fcdOutInfoFileName,
                 "--fcd-output.acceleration"])
    veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error = run(fis_start_time, end_time)
    # convert new xml file to csv

    xml2csv.main([fcdOutInfoFileName])
    xml2csv.main([fullOutFileName])

    fcdOutCSV = os.path.splitext(fcdOutInfoFileName)[0]+'.csv'
    fullOutCSV = os.path.splitext(fullOutFileName)[0]+'.csv'

    print(f"The FCD outfile file was generate: {fcdOutCSV}")

    if options.krauss:
        title = "Krauss"
    else:
        title = "FIS"
    df_FCD = pd.read_csv(f'{fcdOutCSV}')
    df_Full = pd.read_csv(f'{fullOutCSV}')
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

    fullTime0 = []
    fullTime1 = []
    fullTime2 = []
    fullTime3 = []
    fullTime4 = []

    veh0CO2 = []
    veh1CO2 = []
    veh2CO2 = []
    veh3CO2 = []
    veh4CO2 = []

    veh0CO = []
    veh1CO = []
    veh2CO = []
    veh3CO = []
    veh4CO = []

    veh0Fuel = []
    veh1Fuel = []
    veh2Fuel = []
    veh3Fuel = []
    veh4Fuel = []

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

    for index, row in df_Full.iterrows():
        # print(row["vehicle_id"], row["vehicle_pos"])
        if row["vehicle_id"] == 0:
            fullTime0.append(row["data_timestep"])
            veh0CO2.append(row["vehicle_CO2"])
            veh0CO.append(row["vehicle_CO"])
            veh0Fuel.append(row["vehicle_fuel"])
        elif row["vehicle_id"] == 1:
            fullTime1.append(row["data_timestep"])
            veh1CO2.append(row["vehicle_CO2"])
            veh1CO.append(row["vehicle_CO"])
            veh1Fuel.append(row["vehicle_fuel"])
        elif row["vehicle_id"] == 2:
            fullTime2.append(row["data_timestep"])
            veh2CO2.append(row["vehicle_CO2"])
            veh2CO.append(row["vehicle_CO"])
            veh2Fuel.append(row["vehicle_fuel"])
        elif row["vehicle_id"] == 3:
            fullTime3.append(row["data_timestep"])
            veh3CO2.append(row["vehicle_CO2"])
            veh3CO.append(row["vehicle_CO"])
            veh3Fuel.append(row["vehicle_fuel"])
        elif row["vehicle_id"] == 4:
            fullTime4.append(row["data_timestep"])
            veh4CO2.append(row["vehicle_CO2"])
            veh4CO.append(row["vehicle_CO"])
            veh4Fuel.append(row["vehicle_fuel"])

    veh0CO2Sum = sum(veh0CO2)
    veh1CO2Sum = sum(veh1CO2)
    veh2CO2Sum = sum(veh2CO2)
    veh3CO2Sum = sum(veh3CO2)
    veh4CO2Sum = sum(veh4CO2)
    # try:
    #     os.mkdir(f'./{images_subdirectory}/')
    # except Exception:
    #     pass
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(time0, veh0Position, label="Vehicle 0")
    ax.plot(time1, veh1Position, label="Vehicle 1")
    ax.plot(time2, veh2Position, label="Vehicle 2")
    ax.plot(time3, veh3Position, label="Vehicle 3")
    ax.plot(time4, veh4Position, label="Vehicle 4")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Position')
    ax.legend()
    ax.set_title(f"{title} Vehcile Position vs Time")
    posFile = f'./{images_subdirectory}/{title}_vehicle_position.png'
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
    ax.set_title(f"{title} Vehcile Velocity vs Time")
    velFile = f'./{images_subdirectory}/{title}_vehicle_velocity.png'
    if os.path.isfile(velFile):
        os.unlink(velFile)
    fig.savefig(f'{velFile}')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(time0, veh0Acceleration, label="Krauss Lead Vehicle")
    ax.plot(time1, veh1Acceleration, label=f"{title} Follower 1")
    ax.plot(time2, veh2Acceleration, label=f"{title} Follower 2")
    ax.plot(time3, veh3Acceleration, label=f"{title} Follower 2")
    ax.plot(time4, veh4Acceleration, label=f"{title} Follower 2")
    ax.axhline(y=0.93, color='r', linestyle='-', label="Discomfort Threshold")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Acceleration')
    ax.legend()
    ax.set_title(f"{title} Vehcile Acceleration vs Time")
    accelFile = f'./{images_subdirectory}/{title}_vehicle_acceleration.png'
    if os.path.isfile(accelFile):
        os.unlink(accelFile)
    fig.savefig(f'{accelFile}')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(range(len(veh1_gap_error)), veh1_gap_error, label="Vehicle 1")
    ax.plot(range(len(veh2_gap_error)), veh2_gap_error, label="Vehicle 2")
    ax.plot(range(len(veh3_gap_error)), veh3_gap_error, label="Vehicle 3")
    ax.plot(range(len(veh4_gap_error)), veh4_gap_error, label="Vehicle 4")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Gap Error')
    ax.legend()
    ax.set_title(f"{title} Vehcile Gap Error vs Time")
    gapErrFile = f'./{images_subdirectory}/{title}_vehicle_gap.png'
    if os.path.isfile(gapErrFile):
        os.unlink(gapErrFile)
    fig.savefig(f'{gapErrFile}')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.scatter(fullTime0, veh0CO2, label="Krauss Lead Vehicle")
    ax.scatter(fullTime1, veh1CO2, label=f"{title} Follower 1")
    ax.scatter(fullTime2, veh2CO2, label=f"{title} Follower 2")
    ax.scatter(fullTime3, veh3CO2, label=f"{title} Follower 3")
    ax.scatter(fullTime4, veh4CO2, label=f"{title} Follower 4")
    ax.set_xlabel('Time_Step')
    ax.set_ylabel('Vehicle CO2 Emission')
    ax.legend()
    ax.set_title(f"{title} Vehcile CO2 Emission vs Time")
    co2File = f'./{images_subdirectory}/{title}_vehicle_co2.png'
    if os.path.isfile(co2File):
        os.unlink(co2File)
    fig.savefig(f'{co2File}')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.scatter(0, veh0CO2Sum, label="Krauss Lead Vehicle")
    ax.scatter(1, veh1CO2Sum, label=f"{title} Follower 1")
    ax.scatter(2, veh2CO2Sum, label=f"{title} Follower 2")
    ax.scatter(3, veh3CO2Sum, label=f"{title} Follower 3")
    ax.scatter(4, veh4CO2Sum, label=f"{title} Follower 4")
    ax.set_xlabel('Vehicle')
    ax.set_ylabel('Total Vehicle CO2 Emission')
    ax.legend()
    ax.set_title(f"{title} Total Vehcile CO2 Emission")
    co2FileTotal = f'./{images_subdirectory}/{title}_vehicle_total_co2.png'
    if os.path.isfile(co2FileTotal):
        os.unlink(co2FileTotal)
    fig.savefig(f'{co2FileTotal}')
