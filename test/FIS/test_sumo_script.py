#!/usr/bin/env python

import csv
import gc
import matplotlib.pyplot as plt
from memory_profiler import profile
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
sys.path.append(r'C:/Program Files (x86)/Eclipse/Sumo/tools/xml')
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
    opt_parser.add_option("--highway_2", action="store_true",
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
# @profile
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

    veh1_gap_error_rate = []
    veh2_gap_error_rate = []
    veh3_gap_error_rate = []
    veh4_gap_error_rate = []

    veh1_lane_change_decision = []
    veh2_lane_change_decision = []
    veh3_lane_change_decision = []
    veh4_lane_change_decision = []

    totalTimeLoss = []

    TTL = np.empty((0, 4), int)

    GA_chromosome = [[-1.4457460055267017],[0.9596282534579326],[7],[-1.9687233494625112],[-1.4949227177070799],[0.4612586650427106],[-0.5184262059508764],[0.33112194921412064],[1],[0.032862387192658105],[1.1411546493663296],[1.6209736373574386],[-1.325107367543487],[-1.0780860231020188],[-0.5188892088706325],[0.35693070676711347],[2.050684365001913],[2.774034566185253],[1],[2.68280418027796],[2.748702738024731],[2],[3.489405091498025],[5.0337841895636455],[-5.724586988002178],[-0.35251959938796507],[1],[-4.750454037624641],[-4.254990848572678],[7],[1],[3.2731361566159016],[7],[-5.90119953417147],[-3.844035790352814],[3.8775225627050975],[-5.2362274660785495],[-1.168961150316191],[9],[0.23788695355013978],[3.1587191623626643],[4.687913542495231],[1],[7.525508667366361],[7.804668032310376],[1.342919268417213],[1.39218145634533],[1.6151422656498324],[0.31956898205956374],[1.0963867260814184],[2.2544998100268643],[1],[2.490873025143525],[4.1265360934946305],[-0.6069999197682163],[0.6492415564704332],[1],[-0.1372509781448512],[1.347515967623831],[1.3896826810635972],[-3.6379127234475956],[-1.6691021049343568],[0.2178198719024973],[-4.54308179374002],[0.0989252235612108],[1.3340108981072287]]

    i = 0
    new_data = []
    while i < len(GA_chromosome):
        new_data.append(GA_chromosome[i][0])
        i = i+1

    j = 0
    chromosome_array_of_arrays = []
    while j < len(new_data):
        chromosome_array_of_arrays.append(new_data[j:j+3])
        j = j + 3

    # convert array of arrays to a Numpy array of arrays
    chromosome_array_of_arrays = np.array(chromosome_array_of_arrays)

    # hand picked FIS parameters
    membership_function_values = np.array([
                                            [-2, -1, -0.5],
                                            [-0.6, -0.5, -0.25],
                                            [-0.5, -0.25, 0],
                                            [-0.25, 0, 0.25],
                                            [0, 0.5, 1],
                                            [0.5, 1, 1.5],
                                            [1, 1.5, 3],
                                            # second input mem functions
                                            [-10, -7.5, -5.6],
                                            [-6, -5.36, -2.235],
                                            [-5.36, -2.235, -0.447],
                                            [-10, -2.235, 0],
                                            [-0.447, 0, 0.447],
                                            [0, 0.447, 2.235],
                                            [0.447, 2.235, 5.36],
                                            [2.235, 5.36, 10],
                                            # output membership functions
                                            [-5, -4.572, -3],
                                            [-4.572, -3, -1.5],
                                            [-2.235, -1.5, 0],
                                            [-1.5, 0, 1.5],
                                            [0, 1.5, 3],
                                            [1.5, 3, 4.572],
                                            [3, 4.572, 5]
                                            ])

    lane_change_membership_function_values = np.array([
                                            [-1, 0, 15],
                                            [14, 22, 29],
                                            [25, 50, 100],
                                            # second input mem functions
                                            [-0.1, 0, 0.03],
                                            [0.025, 0.05, 0.07],
                                            [0.065, 0.1, 1],
                                            # output membership functions
                                            [-0.1, 0, 0.49],
                                            [0.48, 1, 1.1],
                                            ])

    # hand picked FIS parameters
    second_longitudinal_membership_function_values = np.array([
                                            [-10, -1.5, 0],
                                            [-1.5, 0, 1.5],
                                            [0, 1.5, 10],
                                            # second input mem functions
                                            [-10, -2.235, 0],
                                            [-0.447, 0, 0.447],
                                            [0, 0.447, 2.235],
                                            # output membership functions
                                            [-2.235, -1.5, 0],
                                            [-1.5, 0, 1.5],
                                            [0, 1.5, 3]
                                            ])
    SUMO = fuzzyLogic.createFuzzyControl(membership_function_values)
    SUMOLANECHANGE = fuzzyLogic.createFuzzyLaneControl(lane_change_membership_function_values)
    SUMOSECONDLONGITUDE = fuzzyLogic.createSecondFuzzyLongitudinalControl(second_longitudinal_membership_function_values)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        vehPosition = []
        vehSpeed = []
        newVehPosition = []
        newVehSpeed = []
        timeLoss = []
        newVehTimeLoss = []
        # use the Krauss vehicle controller
        if options.krauss:
            if 30 < step < end_time:
                for ind in traci.vehicle.getIDList():
                    veh5_lane = traci.vehicle.getLaneID("5")
                    # print(veh5_lane)
                    if veh5_lane != 1:
                        traci.vehicle.changeLane("5", 1, 3)
                    if int(ind) < 5:
                        vehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                        vehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                        if int(ind) > 0:
                            timeLoss.append(traci.vehicle.getTimeLoss(f"{ind}"))
                    else:
                        newVehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                        newVehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                        newVehTimeLoss.append(traci.vehicle.getTimeLoss(f"{ind}"))
                traci.vehicle.setSpeed("5", 31)
                veh1Previous_Gap = (vehPosition[0][0] - 5 - vehPosition[1][0]) / vehSpeed[1]  # gap with previous car units: seconds
                veh1_gap_error.append(veh1Previous_Gap-1)
                veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                veh2_gap_error.append(veh2Previous_Gap-1)
                veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                veh3_gap_error.append(veh3Previous_Gap-1)
                veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                veh4_gap_error.append(veh4Previous_Gap-1)

                totalTimeLoss.append(sum(timeLoss))

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
                for ind in traci.vehicle.getIDList():
                    veh5_lane = traci.vehicle.getLaneID("5")
                    # print(veh5_lane)
                    if veh5_lane != 1:
                        traci.vehicle.changeLane("5", 1, 3)
                    if int(ind) < 5:
                        vehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                        vehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                        if int(ind) > 0:
                            timeLoss.append(traci.vehicle.getTimeLoss(f"{ind}"))
                    else:
                        newVehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                        newVehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                        newVehTimeLoss.append(traci.vehicle.getTimeLoss(f"{ind}"))
                veh1_gap_error.append(np.nan)
                veh2_gap_error.append(np.nan)
                veh3_gap_error.append(np.nan)
                veh4_gap_error.append(np.nan)
                totalTimeLoss.append(sum(timeLoss))
        # Use the FIS vehicle controller
        else:
            if 30 < step < fis_start_time + 1:
                for ind in traci.vehicle.getIDList():
                    veh5_lane = traci.vehicle.getLaneID("5")
                    # print(veh5_lane)
                    if veh5_lane != 1:
                        traci.vehicle.changeLane("5", 1, 3)
                    if int(ind) < 5:
                        vehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                        vehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                        if int(ind) > 0:
                            timeLoss.append(traci.vehicle.getTimeLoss(f"{ind}"))
                    else:
                        newVehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                        newVehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                        newVehTimeLoss.append(traci.vehicle.getTimeLoss(f"{ind}"))
                veh1Previous_Gap = (vehPosition[0][0] - 5 - vehPosition[1][0]) / vehSpeed[1]
                veh1_gap_error.append(veh1Previous_Gap-1)
                veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                veh2_gap_error.append(veh2Previous_Gap-1)
                veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                veh3_gap_error.append(veh3Previous_Gap-1)
                veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                veh4_gap_error.append(veh4Previous_Gap-1)
                previousTimeLoss = timeLoss
                totalTimeLoss.append(sum(timeLoss))

                time_to_collision = calculateTimeToCollision(vehSpeed, vehPosition)
                TTL = np.vstack([TTL, np.array(time_to_collision)])

            elif fis_start_time < step < end_time:
                # is re-creating these variables necessary???????
                vehPosition = []
                vehSpeed = []
                vehicleGapErrors = []
                newVehPosition = []
                newVehSpeed = []
                timeLoss = []
                newVehTimeLoss = []
                newLaneVehicles = []
                for ind in traci.vehicle.getIDList():
                    veh5_lane = traci.vehicle.getLaneID("5")
                    # print(veh5_lane)
                    if veh5_lane != 1:
                        traci.vehicle.changeLane("5", 1, 3)
                    if int(ind) < 5:
                        vehPosition.append(traci.vehicle.getPosition(f"{ind}"))
                        vehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                        if int(ind) > 0:
                            timeLoss.append(traci.vehicle.getTimeLoss(f"{ind}"))
                    else:
                        newVehPosition.append(traci.vehicle.getLanePosition(f"{ind}"))
                        newVehSpeed.append(traci.vehicle.getSpeed(f"{ind}"))
                        newVehTimeLoss.append(traci.vehicle.getTimeLoss(f"{ind}"))
                totalTimeLoss.append(sum(timeLoss))
                traci.vehicle.setSpeed("5", 31)
                veh1_lane = traci.vehicle.getLaneID("1")
                veh1_lane_value = int(veh1_lane[6])
                if veh1_lane_value == 1:
                    # print(veh1_lane_value)
                    traci.vehicle.changeLane("1", 1, 300)
                    traci.vehicle.changeLane("2", 1, 300)
                    traci.vehicle.changeLane("3", 1, 300)
                    traci.vehicle.changeLane("4", 1, 300)

                    # avgTimeLoss = sum(timeLoss)/4
                    # timeLossChangeRate = sum([a - b for a, b in zip(timeLoss, previousTimeLoss)])/4

                    veh2 = fuzzyLogic.calc_Inputs(2, vehPosition[1][0], veh2Previous_Gap, vehPosition[2][0], vehSpeed[2], vehicleGapErrors, avgTimeLoss, timeLossChangeRate, SUMO, SUMOLANECHANGE)
                    veh2Previous_Gap = veh2[0]
                    veh2_gap.append(veh2[0])
                    vehicleGapErrors.append(veh2[1])
                    veh2Acceleration = veh2[3]
                    veh2Speed = vehSpeed[2] + veh2Acceleration
                    veh2_gap_error.append(veh2[1])
                    veh2_gap_error_rate.append(veh2[2])
                    veh2_lane_change_decision.append(veh2[4])

                    # if veh1_lane_value == 1:
                    #     traci.vehicle.changeLane("2", 1, 300)

                    traci.vehicle.setSpeed("2", veh2Speed)

                    veh3 = fuzzyLogic.calc_Inputs(3, vehPosition[2][0], veh3Previous_Gap, vehPosition[3][0], vehSpeed[3], vehicleGapErrors, avgTimeLoss, timeLossChangeRate, SUMO, SUMOLANECHANGE)
                    veh3Previous_Gap = veh3[0]
                    veh3_gap.append(veh3[0])
                    vehicleGapErrors.append(veh3[1])
                    veh3Acceleration = veh3[3]
                    veh3Speed = vehSpeed[3] + veh3Acceleration
                    veh3_gap_error.append(veh3[1])
                    veh3_gap_error_rate.append(veh3[2])
                    veh3_lane_change_decision.append(veh3[4])
                    traci.vehicle.setSpeed("3", veh3Speed)

                    # if veh1_lane_value == 1:
                    #     traci.vehicle.changeLane("3", 1, 300)

                    veh4 = fuzzyLogic.calc_Inputs(4, vehPosition[3][0], veh4Previous_Gap, vehPosition[4][0], vehSpeed[4], vehicleGapErrors, avgTimeLoss, timeLossChangeRate, SUMO, SUMOLANECHANGE)
                    veh4Previous_Gap = veh4[0]
                    veh4_gap.append(veh4[0])
                    veh4Acceleration = veh4[3]
                    veh4Speed = vehSpeed[4] + veh4Acceleration
                    veh4_gap_error.append(veh4[1])
                    veh4_gap_error_rate.append(veh4[2])
                    veh4_lane_change_decision.append(veh4[4])
                    traci.vehicle.setSpeed("4", veh4Speed)
                    time_to_collision = calculateTimeToCollision(vehSpeed, vehPosition)

                    veh1Speed = traci.vehicle.getSpeed("1")
                    veh2Speed = traci.vehicle.getSpeed("2")
                    veh3Speed = traci.vehicle.getSpeed("3")
                    veh4Speed = traci.vehicle.getSpeed("4")
                    platoonSpeedDiff = [31.292 - x for x in [veh1Speed, veh2Speed, veh3Speed, veh4Speed]]
                    platoonGapError = [veh2_gap_error[-1], veh3_gap_error[-1], veh4_gap_error[-1]]
                    SUMOSECONDLONGITUDE.input['platoon-gap-error-value'] = (sum(platoonGapError) / len(platoonGapError))
                    SUMOSECONDLONGITUDE.input['vehicle-error-value'] = platoonSpeedDiff[0]
                    SUMOSECONDLONGITUDE.compute()
                    result = SUMOSECONDLONGITUDE.output['acceleration-value']
                    veh1Speed = veh1Speed + result
                    vehicleGapErrors.append(0)
                    veh1_gap_error.append(0)
                    veh1_gap_error_rate.append(0)
                    avgTimeLoss = sum(timeLoss)/4
                    timeLoss.append(traci.vehicle.getTimeLoss("1"))
                    timeLossChangeRate = sum([a - b for a, b in zip(timeLoss, previousTimeLoss)])/4
                    previousTimeLoss = timeLoss
                    traci.vehicle.setSpeed("1", veh1Speed)

                    time_to_collision[0] = 0
                    TTL = np.vstack([TTL, np.array(time_to_collision)])

                else:
                    avgTimeLoss = sum(timeLoss)/4
                    timeLossChangeRate = sum([a - b for a, b in zip(timeLoss, previousTimeLoss)])/4
                    previousTimeLoss = timeLoss
                    veh1 = fuzzyLogic.calc_Inputs(1, vehPosition[0][0], veh1Previous_Gap, vehPosition[1][0], vehSpeed[1], vehicleGapErrors, avgTimeLoss, timeLossChangeRate, SUMO, SUMOLANECHANGE)
                    veh1Previous_Gap = veh1[0]
                    veh1_gap.append(veh1[0])
                    vehicleGapErrors.append(veh1[1])
                    veh1Acceleration = veh1[3]
                    veh1Speed = vehSpeed[1] + veh1Acceleration
                    veh1_gap_error.append(veh1[1])
                    veh1_gap_error_rate.append(veh1[2])
                    veh1_lane_change_decision.append(veh1[4])
                    traci.vehicle.setSpeed("1", veh1Speed)
                    # print(traci.lane.getIDList())
                    # if veh1_lane_change_decision[-1] == 1:

                    # make this a function:
                    # print(vehPosition[0][0], traci.vehicle.getPosition("5")[0])

                    veh2 = fuzzyLogic.calc_Inputs(2, vehPosition[1][0], veh2Previous_Gap, vehPosition[2][0], vehSpeed[2], vehicleGapErrors, avgTimeLoss, timeLossChangeRate, SUMO, SUMOLANECHANGE)
                    veh2Previous_Gap = veh2[0]
                    veh2_gap.append(veh2[0])
                    vehicleGapErrors.append(veh2[1])
                    veh2Acceleration = veh2[3]
                    veh2Speed = vehSpeed[2] + veh2Acceleration
                    veh2_gap_error.append(veh2[1])
                    veh2_gap_error_rate.append(veh2[2])
                    veh2_lane_change_decision.append(veh2[4])

                    # if veh1_lane_value == 1:
                    #     traci.vehicle.changeLane("2", 1, 300)

                    traci.vehicle.setSpeed("2", veh2Speed)

                    veh3 = fuzzyLogic.calc_Inputs(3, vehPosition[2][0], veh3Previous_Gap, vehPosition[3][0], vehSpeed[3], vehicleGapErrors, avgTimeLoss, timeLossChangeRate, SUMO, SUMOLANECHANGE)
                    veh3Previous_Gap = veh3[0]
                    veh3_gap.append(veh3[0])
                    vehicleGapErrors.append(veh3[1])
                    veh3Acceleration = veh3[3]
                    veh3Speed = vehSpeed[3] + veh3Acceleration
                    veh3_gap_error.append(veh3[1])
                    veh3_gap_error_rate.append(veh3[2])
                    veh3_lane_change_decision.append(veh3[4])
                    traci.vehicle.setSpeed("3", veh3Speed)

                    # if veh1_lane_value == 1:
                    #     traci.vehicle.changeLane("3", 1, 300)

                    veh4 = fuzzyLogic.calc_Inputs(4, vehPosition[3][0], veh4Previous_Gap, vehPosition[4][0], vehSpeed[4], vehicleGapErrors, avgTimeLoss, timeLossChangeRate, SUMO, SUMOLANECHANGE)
                    veh4Previous_Gap = veh4[0]
                    veh4_gap.append(veh4[0])
                    veh4Acceleration = veh4[3]
                    veh4Speed = vehSpeed[4] + veh4Acceleration
                    veh4_gap_error.append(veh4[1])
                    veh4_gap_error_rate.append(veh4[2])
                    veh4_lane_change_decision.append(veh4[4])
                    traci.vehicle.setSpeed("4", veh4Speed)

                    traci.vehicle.changeLane("1", 0, 300)
                    traci.vehicle.changeLane("2", 0, 300)
                    traci.vehicle.changeLane("3", 0, 300)
                    traci.vehicle.changeLane("4", 0, 300)

                    # print(veh1_lane_change_decision[-1], veh2_lane_change_decision[-1], veh3_lane_change_decision[-1], veh4_lane_change_decision[-1])

                    # if veh1_lane_value == 1:
                    #     traci.vehicle.changeLane("4", 1, 300)
                    if veh1_lane_change_decision[-1] and veh2_lane_change_decision[-1] and veh3_lane_change_decision[-1] and veh4_lane_change_decision[-1] == 1:  # step > 900:
                        veh_gap_error_max = max([veh1_gap_error[-1], veh2_gap_error[-1], veh3_gap_error[-1], veh4_gap_error[-1]])
                        if veh_gap_error_max < 0.5:
                            # now engage the last FIS
                            # determine if there is enough room to change lanes
                            newLaneDistanceDiff = [vehPosition[1][0] - traci.vehicle.getPosition("5")[0], vehPosition[2][0] - traci.vehicle.getPosition("5")[0], vehPosition[3][0] - traci.vehicle.getPosition("5")[0], vehPosition[4][0] - traci.vehicle.getPosition("5")[0]]
                            if all([abs(x) > 250 for x in newLaneDistanceDiff]):
                                # print(veh1_lane_change_decision)
                                traci.vehicle.changeLane("1", 1, 300)
                                traci.vehicle.changeLane("2", 1, 300)
                                traci.vehicle.changeLane("3", 1, 300)
                                traci.vehicle.changeLane("4", 1, 300)
                                # print(traci.vehicle.getLaneID("1"))

                    time_to_collision = calculateTimeToCollision(vehSpeed, vehPosition)
                    TTL = np.vstack([TTL, np.array(time_to_collision)])

                # traci.vehicle.changeLane("1", 1, 2)

                # del veh1, veh2, veh3, veh4, vehSpeed, vehPosition
                # gc.collect()

                if options.slow_down_midway:
                    if 525 < step < 615:
                        traci.vehicle.slowDown("0", 10, 90)
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
    # del fuzzyLogic
    # gc.collect()
    return veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error, veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, TTL, totalTimeLoss


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
        sumoBinary = checkBinary('sumo-gui')
    fileName = ntpath.basename(__file__).split('.')[0]

    highway_filename = "highway_2"
    fileName_no_sffix = f"{fileName}_{highway_filename}"
    fis_start_time = 300
    end_time = 2000

    timestr = time.strftime("%Y%m%d_%S")

    # create *_subdirectory or join it
    try:
        os.mkdir("./results/spreadsheet")
    except Exception:
        pass

    try:
        os.mkdir("./results/images")
    except Exception:
        pass

    spreadsheet_subdirectory = f"./results/spreadsheet/{timestr}_{fileName_no_sffix}_tripInfo"
    global images_subdirectory
    images_subdirectory = f"./results/images/{timestr}_{fileName_no_sffix}_tripInfo"

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
    while os.path.exists(os.path.join(spreadsheet_subdirectory, f"{i}_fcdout.xml")):
        i += 1
    recnum = format(int(i), '03d')
    # another way to seperate new log files:
    # https://sumo.dlr.de/docs/Simulation/Output/index.html#separating_outputs_of_repeated_runs

    # generate route file
    routeFileName = f"{highway_filename}.rou.xml"
    # inductionLoopFileName = "{}_induction.xml".format(recnum)

    # generate additional file
    additionalFileName = f"{highway_filename}.add.xml"
    # inductionLoopFileName = f"./results/{recnum}_induction.xml"
    # generate_additionalfile(additionalFileName, inductionLoopFileName)

    ssmFileName = rf"{spreadsheet_subdirectory}\{recnum}_ssm.xml"
    fullOutFileName = rf"{spreadsheet_subdirectory}\{recnum}_fullout.xml"
    fcdOutInfoFileName = rf"{spreadsheet_subdirectory}\{recnum}_fcdout.xml"
    amitranInforFileName = rf"{spreadsheet_subdirectory}\{recnum}_amitran.xml"
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", f"{highway_filename}.sumocfg",
                "--route-files", routeFileName,
                 "--additional-files", additionalFileName,
                 "--device.ssm.probability", "1",
                 "--device.ssm.file", ssmFileName,
                 "--full-output", fullOutFileName,
                 "--fcd-output", fcdOutInfoFileName,
                 "--fcd-output.acceleration",
                 "--start",
                 "--quit-on-end"])
    # call the run script. Runs the fuzzy logic
    veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error, \
        veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, TTL, totalTimeLoss = run(fis_start_time, end_time)

    veh1_fitness_sum = sum(veh1_gap_error[fis_start_time:end_time])
    veh2_fitness_sum = sum(veh2_gap_error[fis_start_time:end_time])
    veh3_fitness_sum = sum(veh3_gap_error[fis_start_time:end_time])
    veh4_fitness_sum = sum(veh4_gap_error[fis_start_time:end_time])

    fitness = np.sum([veh1_fitness_sum, veh2_fitness_sum, veh3_fitness_sum, veh4_fitness_sum])

    print(f"The fitness is: {fitness}")
    print(f"The total time lost was: {sum(totalTimeLoss)}")

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
    df_FCD = pd.read_csv(fcdOutCSV)  # './results/spreadsheet/20230306_highway_1_tripInfo/000_fcdout.csv')
    df_Full = pd.read_csv(fullOutCSV)  # './results/spreadsheet/20230306_highway_1_tripInfo/000_fullout.csv')
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
