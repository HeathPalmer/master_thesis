#!/usr/bin/env python
import csv
from EasyGA import GA, crossover
import matplotlib.pyplot as plt
import ntpath
import numpy as np
import os
import optparse
import pandas as pd
import random
import sys
import time
import traci
from itertools import chain
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


class My_GA(GA):
    """Example GA with custom initialize_population method."""

    def create_pop(self):
        membershipFunction = []
        for x in range(7):
            outterBounds = [random.uniform(-2, 3), random.uniform(-2, 3)]
            outterBounds.sort()
            peak = random.uniform(outterBounds[0], outterBounds[1])
            membershipFunction.append(outterBounds[0])
            membershipFunction.append(peak)
            membershipFunction.append(outterBounds[1])

        for x in range(8):
            outterBounds = [random.uniform(-6, 10), random.uniform(-6, 10)]
            outterBounds.sort()
            peak = random.uniform(outterBounds[0], outterBounds[1])
            membershipFunction.append(outterBounds[0])
            membershipFunction.append(peak)
            membershipFunction.append(outterBounds[1])

        for x in range(7):
            outterBounds = [random.uniform(-5, 5.1), random.uniform(-5, 5.1)]
            outterBounds.sort()
            peak = random.uniform(outterBounds[0], outterBounds[1])
            membershipFunction.append(outterBounds[0])
            membershipFunction.append(peak)
            membershipFunction.append(outterBounds[1])

        return membershipFunction

    def initialize_population(self):
        """Custom population initialization with chromosomes
        """
        ga_test = My_GA()

        self.population = self.make_population(
            ga_test.create_pop()
            for _
            in range(self.population_size)
        )
        print(f"The population is: {self.population}")


# fitness function
def user_def_fitness(chromosome):
    try:
        print(f"The solution was: {chromosome}")
        # flat_solution = list(chain.from_iterable(solution))
        # print(flat_solution)
        i = 0
        new_data = []
        while i < len(chromosome):
            new_data.append(chromosome[i].value)
            i = i+1

        j = 0
        chromosome_array_of_arrays = []
        while j < len(new_data):
            chromosome_array_of_arrays.append(new_data[j:j+3])
            j = j + 3

        # convert array of arrays to a Numpy array of arrays
        chromosome_array_of_arrays = np.array(chromosome_array_of_arrays)
        # print(f"The proposed chromosome is {chromosome_array_of_arrays}")

        # starting SUMO and traci

        global fullOutFileName, fcdOutInfoFileName, recnum
        # set the file name based on increamenting value
        i = 0
        while os.path.exists(os.path.join(spreadsheet_subdirectory, "%s_fcdout.xml" % format(int(i), '03d'))):
            i += 1
        recnum = format(int(i), '03d')
        ssmFileName = rf"{spreadsheet_subdirectory}\{recnum}_ssm.xml"
        fullOutFileName = rf"{spreadsheet_subdirectory}\{recnum}_fullout.xml"
        fcdOutInfoFileName = rf"{spreadsheet_subdirectory}\{recnum}_fcdout.xml"

        # amitranInforFileName = rf"{spreadsheet_subdirectory}\{recnum}_amitran.xml"
        # traci starts sumo as a subprocess and then this script connects and runs
        traci.start([sumoBinary, "-c", f"{fileName_No_Suffix}.sumocfg",
                     "--route-files", routeFileName,
                     "--additional-files", additionalFileName,
                     "--device.ssm.probability", "1",
                     "--device.ssm.file", ssmFileName,
                     "--full-output", fullOutFileName,
                     "--fcd-output", fcdOutInfoFileName,
                     "--fcd-output.acceleration"])

        global veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error, \
            veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, TTL

        # # check if the proposed solution is valid...
        # for membership_function_row in new_data:
        #     if membership_function_row[0] > membership_function_row[1] > membership_function_row[2]:
        #         memberships_acceptable = True
        #     else:
        #         print("The membership function bounds are not acceptable")
        #         memberships_acceptable = False

        # if memberships_acceptable:
        # run the test_sumo_script
        veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error, \
            veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, TTL = run(fis_start_time, end_time, chromosome_array_of_arrays)

        veh1_fitness_sum = sum(veh1_gap_error[fis_start_time:end_time])
        veh2_fitness_sum = sum(veh2_gap_error[fis_start_time:end_time])
        veh3_fitness_sum = sum(veh3_gap_error[fis_start_time:end_time])
        veh4_fitness_sum = sum(veh4_gap_error[fis_start_time:end_time])

        fitness = np.sum([veh1_fitness_sum, veh2_fitness_sum, veh3_fitness_sum, veh4_fitness_sum])

            # total error for the simulation
            # fitness = np.sum([veh1_fitness_sum, veh2_fitness_sum, veh3_fitness_sum, veh4_fitness_sum])
        # else:
        #     fitness = 10000000  # one million

    except Exception as e:
        # fitness = np.sum([veh1_fitness_sum, veh2_fitness_sum, veh3_fitness_sum, veh4_fitness_sum])
        fitness = 10000000
        print(f"There was an error calculating the fitness. The error was: {e}")
    finally:
        print("Attempted to run fitness")
        return fitness


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


def calculateTimeToCollision(vehSpeed, vehPosition):  # gap_distance - units: meters
    veh_speed_diff = []
    for indx, speed in enumerate(vehSpeed[1:]):
        veh_speed_diff.append(vehSpeed[indx] - vehSpeed[indx-1])

    time_to_collision = []
    for indx, speed_diff in enumerate(veh_speed_diff):
        # print(indx)
        if speed_diff > 0:
            gap_distance = vehPosition[indx][0] - 5 - vehPosition[indx+1][0]
            time_to_collision.append(gap_distance / veh_speed_diff[indx])
        else:
            time_to_collision.append(np.nan)

    return time_to_collision


# contains TraCI control loop
def run(fis_start_time, end_time, chromosome_array_of_arrays):
    try:
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

        veh1_gap_error_rate = []
        veh2_gap_error_rate = []
        veh3_gap_error_rate = []
        veh4_gap_error_rate = []

        TTL = np.empty((0, 4), int)

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            vehPosition = []
            vehSpeed = []
            # use the Krauss vehicle controller
            if options.krauss:
                if 30 < step < end_time:
                    for ind in traci.vehicle.getIDList():
                        vehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                        vehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                    veh1Previous_Gap = (vehPosition[0][0] - 5 - vehPosition[1][0]) / vehSpeed[1]  # gap with previous car units: seconds
                    veh1_gap_error.append(veh1Previous_Gap-1)
                    veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                    veh2_gap_error.append(veh2Previous_Gap-1)
                    veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                    veh3_gap_error.append(veh3Previous_Gap-1)
                    veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                    veh4_gap_error.append(veh4Previous_Gap-1)

                    time_to_collision = calculateTimeToCollision(vehSpeed, vehPosition)
                    TTL = np.vstack([TTL, np.array(time_to_collision)])

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
            # Use the FIS vehicle controller
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

                    time_to_collision = calculateTimeToCollision(vehSpeed, vehPosition)
                    TTL = np.vstack([TTL, np.array(time_to_collision)])

                elif fis_start_time < step < end_time:
                    vehPosition = []
                    vehSpeed = []
                    vehicleGapErrors = []
                    for ind in traci.vehicle.getIDList():
                        vehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                        vehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))

                    veh1 = fuzzyLogic.calc_Inputs(1, vehPosition[0][0], veh1Previous_Gap, vehPosition[1][0], vehSpeed[1], vehicleGapErrors, chromosome_array_of_arrays)
                    veh1Previous_Gap = veh1[0]
                    veh1_gap.append(veh1[0])
                    vehicleGapErrors.append(veh1[1])
                    veh1Acceleration = veh1[3]
                    veh1Speed = vehSpeed[1] + veh1Acceleration
                    veh1_gap_error.append(veh1[1])
                    veh1_gap_error_rate.append(veh1[2])
                    traci.vehicle.setSpeed("1", veh1Speed)

                    veh2 = fuzzyLogic.calc_Inputs(2, vehPosition[1][0], veh2Previous_Gap, vehPosition[2][0], vehSpeed[2], vehicleGapErrors, chromosome_array_of_arrays)
                    veh2Previous_Gap = veh2[0]
                    veh2_gap.append(veh2[0])
                    vehicleGapErrors.append(veh2[1])
                    veh2Acceleration = veh2[3]
                    veh2Speed = vehSpeed[2] + veh2Acceleration
                    veh2_gap_error.append(veh2[1])
                    veh2_gap_error_rate.append(veh2[2])
                    traci.vehicle.setSpeed("2", veh2Speed)

                    veh3 = fuzzyLogic.calc_Inputs(3, vehPosition[2][0], veh3Previous_Gap, vehPosition[3][0], vehSpeed[3], vehicleGapErrors, chromosome_array_of_arrays)
                    veh3Previous_Gap = veh3[0]
                    veh3_gap.append(veh3[0])
                    vehicleGapErrors.append(veh3[1])
                    veh3Acceleration = veh3[3]
                    veh3Speed = vehSpeed[3] + veh3Acceleration
                    veh3_gap_error.append(veh3[1])
                    veh3_gap_error_rate.append(veh3[2])
                    traci.vehicle.setSpeed("3", veh3Speed)

                    veh4 = fuzzyLogic.calc_Inputs(4, vehPosition[3][0], veh4Previous_Gap, vehPosition[4][0], vehSpeed[4], vehicleGapErrors, chromosome_array_of_arrays)
                    veh4Previous_Gap = veh4[0]
                    veh4_gap.append(veh4[0])
                    veh4Acceleration = veh4[3]
                    veh4Speed = vehSpeed[4] + veh4Acceleration
                    veh4_gap_error.append(veh4[1])
                    veh4_gap_error_rate.append(veh4[2])
                    traci.vehicle.setSpeed("4", veh4Speed)

                    time_to_collision = calculateTimeToCollision(vehSpeed, vehPosition)
                    TTL = np.vstack([TTL, np.array(time_to_collision)])

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
    except Exception as e:
        print(f"Error in the FIS run function. Error was {e}")
        step_difference = 1064 - step
        i = 0
        for i in len(step_difference):
            veh1_gap_error.append(10)
            veh2_gap_error.append(10)
            veh3_gap_error.append(10)
            veh4_gap_error.append(10)

            veh1_gap_error_rate.append(10)
            veh2_gap_error_rate.append(10)
            veh3_gap_error_rate.append(10)
            veh4_gap_error_rate.append(10)

            TTL.append(10)

            i = i+1
    finally:
        traci.close()
        sys.stdout.flush()
        return veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error, \
            veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, TTL


def plotResults(x, y, title, xLabel, yLabel, modelType, *plotModification):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    xLength = len(x)
    for i in range(xLength):
        if i == 0:
            if yLabel == "Gap_Error" or yLabel == "Gap_Error_Rate" or yLabel == "TTL_seconds":
                ax.plot(x[i], y[i], label=f"Follower {modelType} Vehicle")
            else:
                ax.plot(x[i], y[i], label="Lead Krauss Vehicle")
        else:
            ax.plot(x[i], y[i], label=f"Follower {modelType} Vehicle {i}")
    if plotModification:
        exec(plotModification[0])
    else:
        pass
    ax.set_xlabel(f'{xLabel}')
    ax.set_ylabel(f'{yLabel}')
    if yLabel == "Jerk":
        ax.legend(loc='upper right')
    else:
        ax.legend()
    ax.set_title(f"{title} Vehcile {yLabel} vs {xLabel}")
    lowerYLabel = yLabel.lower()
    posFile = f'./{images_subdirectory}/{title}_vehicle_{lowerYLabel}.png'
    if os.path.isfile(posFile):
        os.unlink(posFile)
    fig.savefig(f'{posFile}')


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
        sumoBinary = checkBinary('sumo')
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
    global images_subdirectory
    images_subdirectory = f"./results/images/{timestr}_{fileName_No_Suffix}_tripInfo"

    try:
        os.mkdir(spreadsheet_subdirectory)
    except Exception:
        pass

    try:
        os.mkdir(images_subdirectory)
    except Exception:
        pass

    # another way to seperate new log files:
    # https://sumo.dlr.de/docs/Simulation/Output/index.html#separating_outputs_of_repeated_runs

    # generate route file
    routeFileName = f"{fileName_No_Suffix}.rou.xml"
    # inductionLoopFileName = "{}_induction.xml".format(recnum)

    # generate additional file
    additionalFileName = f"{fileName_No_Suffix}.add.xml"
    # inductionLoopFileName = f"./results/{recnum}_induction.xml"
    # generate_additionalfile(additionalFileName, inductionLoopFileName)

    # call the run script. Runs the fuzzy logic
    # veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error, \
    #     veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, TTL = run(fis_start_time, end_time)
    # convert new xml file to csv

    # call the GA
    # Create the Genetic algorithm
    ga = My_GA()

    ga.generation_goal = 10
    ga.chromosome_length = 66
    ga.population_size = 20
    ga.fitness_goal = 0
    # Run everything.
    ga.crossover_individual_impl = crossover.Crossover.Individual.single_point
    ga.fitness_function_impl = user_def_fitness
    # ga.target_fitness_type = 'min'
    ga.evolve()
    # print(ga.population)

    ga.graph.generation_total_fitness()
    ga.graph.show()

    # take the best individual and run the FIS with it again. Then generate the fcOut and so on...

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

    veh0Jerk = []
    veh1Jerk = []
    veh2Jerk = []
    veh3Jerk = []
    veh4Jerk = []

    for index, row in df_FCD.iterrows():
        # print(row["vehicle_id"], row["vehicle_pos"])
        if row["vehicle_id"] == 0:
            time0.append(row["timestep_time"])
            veh0Position.append(row["vehicle_x"])
            veh0Velocity.append(row["vehicle_speed"])
            veh0Acceleration.append(row["vehicle_acceleration"])
            # if fis_start_time < row["timestep_time"] <= end_time:
            if len(veh0Acceleration) > 2:
                acceleration_array_length = len(veh0Acceleration)
                previous_acceleration_value = veh0Acceleration[acceleration_array_length - 2]
                veh0Jerk.append(row["vehicle_acceleration"] - previous_acceleration_value)
            else:
                pass
        elif row["vehicle_id"] == 1:
            time1.append(row["timestep_time"])
            veh1Position.append(row["vehicle_x"])
            veh1Velocity.append(row["vehicle_speed"])
            veh1Acceleration.append(row["vehicle_acceleration"])
            # if fis_start_time < row["timestep_time"] <= end_time:
            if len(veh1Acceleration) > 2:
                acceleration_array_length = len(veh1Acceleration)
                previous_acceleration_value = veh1Acceleration[acceleration_array_length - 2]
                veh1Jerk.append(row["vehicle_acceleration"] - previous_acceleration_value)
        elif row["vehicle_id"] == 2:
            time2.append(row["timestep_time"])
            veh2Position.append(row["vehicle_x"])
            veh2Velocity.append(row["vehicle_speed"])
            veh2Acceleration.append(row["vehicle_acceleration"])
            # if fis_start_time < row["timestep_time"] <= end_time:
            if len(veh2Acceleration) > 2:
                acceleration_array_length = len(veh2Acceleration)
                previous_acceleration_value = veh2Acceleration[acceleration_array_length - 2]
                veh2Jerk.append(row["vehicle_acceleration"] - previous_acceleration_value)
        elif row["vehicle_id"] == 3:
            time3.append(row["timestep_time"])
            veh3Position.append(row["vehicle_x"])
            veh3Velocity.append(row["vehicle_speed"])
            veh3Acceleration.append(row["vehicle_acceleration"])
            # if fis_start_time < row["timestep_time"] <= end_time:
            if len(veh3Acceleration) > 2:
                acceleration_array_length = len(veh3Acceleration)
                previous_acceleration_value = veh3Acceleration[acceleration_array_length - 2]
                veh3Jerk.append(row["vehicle_acceleration"] - previous_acceleration_value)
        elif row["vehicle_id"] == 4:
            time4.append(row["timestep_time"])
            veh4Position.append(row["vehicle_x"])
            veh4Velocity.append(row["vehicle_speed"])
            veh4Acceleration.append(row["vehicle_acceleration"])
            # if fis_start_time < row["timestep_time"] <= end_time:
            if len(veh4Acceleration) > 2:
                acceleration_array_length = len(veh4Acceleration)
                previous_acceleration_value = veh4Acceleration[acceleration_array_length - 2]
                veh4Jerk.append(row["vehicle_acceleration"] - previous_acceleration_value)

    # try:
    #     os.mkdir(f'./{images_subdirectory}/')
    # except Exception:
    #     pass
    x = [time0, time1, time2, time3, time4]
    yPosition = [veh0Position, veh1Position, veh2Position, veh3Position, veh4Position]
    yVelocity = [veh0Velocity, veh1Velocity, veh2Velocity, veh3Velocity, veh4Velocity]
    yAcceleration = [veh0Acceleration, veh1Acceleration, veh2Acceleration, veh3Acceleration, veh4Acceleration]

    xJerk = [range(len(veh0Jerk)), range(len(veh1Jerk)), range(len(veh2Jerk)), range(len(veh3Jerk)), range(len(veh4Jerk))]
    yJerkCalculation = [veh0Jerk, veh1Jerk, veh2Jerk, veh3Jerk, veh4Jerk]

    xGapError = [range(len(veh1_gap_error)), range(len(veh2_gap_error)), range(len(veh3_gap_error)), range(len(veh4_gap_error))]
    yGapError = [veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error]

    xGapErrorRate = [range(len(veh1_gap_error_rate)), range(len(veh2_gap_error_rate)), range(len(veh3_gap_error_rate)), range(len(veh4_gap_error_rate))]
    yGapErrorRate = [veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate]

    xTTL = [range(len(TTL[:, 0])), range(len(TTL[:, 1])), range(len(TTL[:, 2])), range(len(TTL[:, 3]))]
    yTTL = [TTL[:, 0], TTL[:, 1], TTL[:, 2], TTL[:, 3]]

    xGapErrorTranspose = list(zip(*xGapError))
    yGapErrorTranspose = list(zip(*yGapError))
    yGapErrorRateTranspose = list(zip(*yGapErrorRate))
    titleGapError = ["Vehicle 1 Gap Error", "Vehicle 2 Gap Error", "Vehicle 3 Gap Error", "Vehicle 4 Gap Error"]
    titleGapErrorRate = ["Vehicle 1 Gap Error Rate", "Vehicle 2 Gap Error Rate", "Vehicle 3 Gap Error Rate", "Vehicle 4 Gap Error Rate"]

    with open(f'{spreadsheet_subdirectory}/{title}_Gap_error.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(titleGapError)
        csv_writer.writerows(yGapErrorTranspose)

    with open(f'{spreadsheet_subdirectory}/{title}_Gap_error_rate.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(titleGapErrorRate)
        csv_writer.writerows(yGapErrorRateTranspose)

    plotResults(x, yPosition, title, 'Time_Step', 'Position', title)

    plotResults(x, yVelocity, title, 'Time_Step', 'Velocity', title)

    modAcceleration = "ax.axhline(y = 2, color = 'r', linestyle = '-')"
    plotResults(x, yAcceleration, title, 'Time_Step', 'Acceleration', title, str(modAcceleration))

    plotResults(xJerk, yJerkCalculation, title, 'Time_Step', 'Jerk', title)

    # modTTL = "ax.ticklabel_format(axis = \"y\", style = \"sci\", scilimits=(0,2))"
    plotResults(xTTL, yTTL, title, 'Time_Step', 'TTL_seconds', title)

    plotResults(xGapError, yGapError, title, 'Time_Step', 'Gap_Error', title)

    plotResults(xGapErrorRate, yGapErrorRate, title, 'Time_Step', 'Gap_Error_Rate', title)

    # plotResults(xGapError, yGapError, title, 'Time_Step', 'Gap_Error', title)
