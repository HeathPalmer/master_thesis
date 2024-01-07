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
    opt_parser.add_option("--CACC", action="store_true",
                          default=False, help="run the simulation using the CACC model")
    opt_parser.add_option("--GFS", action="store_true",
                          default=False, help="control platoon with GFS controllers instead of FIS controllers")
    opt_parser.add_option("--highway_2_mod", action="store_true",
                          default=False, help="run the simulation using the modified second highway configuration")
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


def parseGAChrmosome(GA_chromosome):
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
    return chromosome_array_of_arrays


def check_Lane_Change(list):
    return all(i[6] == '1' for i in list)


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

    laneChangeDecision = []

    TTL = np.empty((0, 4), int)

    if options.GFS:
        # OLD GA_chromosome = [[-1.6240344751828966],[0.8735664638544858],[2.786421006981838],[-1.5360056631852324],[-1.4109936691129998],[2.7760919030749687],[-1.5146609995028424],[0.286111356789177],[0.533014367315769],[2.5437503030955755],[2.681878309479083],[2.7097058149347957],[-1.1954826254870359],[0.34595014308802696],[7],[0.30548234324726],[1.3912850561655268],[5],[1.9672298354813358],[2.271725813193616],[2.8039497198684],[6.516775246756547],[7.86365500161472],[9],[-3.3358267954019674],[1],[8],[-3.069388543029344],[-0.776325465066936],[9.923128903319997],[0.40372338238102223],[1.2133295148870922],[9.338581482690811],[-3.5131368262457343],[-1.0635482340216882],[-0.8120053202249782],[-2.79827706849043],[4.286758833890948],[8.579088170091405],[-5.114797183793154],[0.3113213534794914],[1.5156892599137048],[-5.106525856456598],[1.3213374066030035],[3.58042381890575],[1.8886598151326774],[2.086456315410275],[4.106743549010657],[-4.581403448687182],[-2.1445655539246036],[-1.6059296614388434],[3.4163387350245618],[4.055041509477416],[4.252104769272686],[2.5453547664446967],[3.3785651221906745],[4.104506478012013],[0.1904316537942723],[1.1529647268301189],[1.6532473842039472],[-1.586099831835674],[-1.4449729751873748],[-0.8071520067124034],[4.209897506678502],[4.537169733825274],[4.714573428575184]]
        GA_chromosome = [[-0.14079584097023234],[0.11270081228641576],[2.9766488665849726],[-0.7968246034900783],[0.7063117908380419],[1.1375978923063292],[2.6074376309201206],[2.9412376263955267],[2.9943432067646683],[0.09504768232433936],[0.1783701401248904],[6],[-1.705442967181982],[-1.0755935953680993],[1.1132247258679815],[1.8814937645809588],[1.9983433314264147],[2.421543784262572],[-1.8739282026849529],[0.3424733263199109],[1.0206152568947475],[-9.197623285153119],[-0.4356347982532265],[2.1123564387579137],[3.6386075378401745],[9.444311177890963],[9.753319640637919],[2],[3.3510412049248295],[8],[1.632166411796124],[3.6923293700206608],[9],[-3.620347050784474],[-1.0357009224151548],[6.252244724018345],[-4.983824480026183],[-2.7087125822618345],[-2.446583149899718],[6.314765767088691],[6.53597866764848],[7.410875527069477],[-1.2412528332455022],[-0.3969996523102406],[10],[-0.06018472251536622],[2.6648896634062136],[3.7646575939607985],[0.4103162471225561],[4.617193296051192],[4.9875776001239025],[-0.8567406719965458],[-0.6112740471510157],[3.3634775190872492],[-0.8014949359170229],[0.4804838856359315],[0.8527166292513932],[-0.31638050827810194],[1.0145093302035575],[2.713314107513235],[-4.086329845081814],[-3.147138561916708],[-1.7753375041178239],[-0.6275491648423106],[2.8023001796869735],[3.2841258840333367]]
        # lat_GA_Chromosome = [[7],[71.8232712481126],[73.44574771934484],[9],[42.97549278960961],[88.51370938381592],[26.480593113993702],[34.333598530390944],[97.87627131835028],[-0.08994987703760998],[0.003994405618960042],[0.6124129233504257],[0.12034402394596203],[0.23316119394964663],[0.29640097938129517],[0.2391229754559647],[0.6980126528742496],[0.6994915574987353],[0.2909558499914311],[0.33025519650852836],[0.3716297699772756],[-0.026500933890887646],[0.8560407770469973],[10]]
        # alt_lat_GA_Chromosome = [[67.6761221512281],[69.83251654757844],[77.76368252882389],[33.97062143232918],[36.332680373664786],[58.947637103181755],[-0.665875480807002],[5.020935566853341],[84.5102309275504],[0.1295196815633867],[0.4524747405064238],[0.5304257536123267],[-0.044061054847725746],[0.4261034949192127],[0.9703554076610142],[0.510278503691667],[0.6033899496234578],[0.6261256068999578],[0.048885670430450806],[0.25171726717588283],[0.35163550672256294],[0.5917424400418568],[0.6148280065175284],[0.6516127105888627]]
        alt_lat_GA_Chromosome = [[64.5156456435981],[65.10512228655486],[86.17156792350178],[1.5577564837304148],[22.647222415305365],[61.19444125696693],[55.64033218802219],[68.99877267290137],[97.42463342627312],[0.11867881846598133],[0.1267105725894678],[0.13111946765751606],[-0.07648265710199151],[0.18819996154948335],[0.8763376540835236],[-0.0492715826292624],[0.33508165688740355],[0.6119936061062785],[0.3242773308948389],[0.39313456372966255],[1.0730835480996557],[0.8863730287887847],[0.8995946312096776],[0.9170835065528654]]
        # OLD lon2_GA_Chromosome = [[1.5098238860638276],[7],[7.92916396487642],[-4.341434003061262],[8.323253529177837],[8.437866725066886],[-0.45026512051033585],[2.719697720047667],[3.528034862358293],[-1.9281714608667464],[4.649377056952073],[8.444484017261313],[-4.810208847723389],[0.5342624851631741],[8],[-5.344379261896522],[-0.189868052993317],[2],[-3.7325788876367287],[3],[5],[-2.634533342311523],[2.4345164558740184],[8],[-3.3872697156344653],[-1.8550680999661042],[1]]
        lon2_GA_Chromosome = [[0.4521444205839842],[3.680796505941435],[6.697808956477672],[-1.8808150437979556],[1.9656094335819412],[4.291756239289974],[-9.539676389386313],[-6.680852310038657],[-3.8071245707475],[-23.96753843201363],[4.171925408735696],[6.530921501997575],[-6.126570272365562],[-1.6231834676335009],[5.321295326278289],[-4.8691723881227915],[-2.0331603647942917],[5.887849324854059],[1.3698664901894961],[4.239744966068314],[4.483935269160289],[-3.505567840175244],[0.24126394521132433],[2.3408711737956835],[-2.68860110830799],[-1.8173721236563323],[-0.8851835663367584]]
        first_longitudinal_membership_function_values = parseGAChrmosome(GA_chromosome)
        # lat_GA_Chromosome = parseGAChrmosome(lat_GA_Chromosome)
        lane_change_membership_function_values = parseGAChrmosome(alt_lat_GA_Chromosome)
        second_longitudinal_membership_function_values = parseGAChrmosome(lon2_GA_Chromosome)
    else:
        # hand picked FIS parameters
        first_longitudinal_membership_function_values = np.array([
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
                                                [-31.292, -2.235, 0],
                                                [-0.447, 0, 0.447],
                                                [0, 0.447, 3],
                                                # output membership functions
                                                [-3, -1.5, 0],
                                                [-1.5, 0, 1.5],
                                                [0, 1.5, 3]
                                                ])

    SUMO = fuzzyLogic.createFuzzyControl(first_longitudinal_membership_function_values)
    SUMOLANECHANGE = fuzzyLogic.createFuzzyLaneControl(lane_change_membership_function_values)
    SUMOSECONDLONGITUDE = fuzzyLogic.createSecondFuzzyLongitudinalControl(second_longitudinal_membership_function_values)
    veh1_lane_state = False

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
                veh1_lane = traci.vehicle.getLaneID("1")
                veh1_lane_value = int(veh1_lane[6])
                if veh1_lane_value == 1:
                    veh1_lane_state = True
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
                            traci.vehicle.setSpeed("5", 31)
                    veh1_gap_error.append(0)
                    veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                    veh2_gap_error.append(veh2Previous_Gap-1)
                    veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                    veh3_gap_error.append(veh3Previous_Gap-1)
                    veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                    veh4_gap_error.append(veh4Previous_Gap-1)

                    totalTimeLoss.append(sum(timeLoss))
                elif veh1_lane_state:
                    veh1_lane_state = True
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
                            traci.vehicle.setSpeed("5", 31)
                    veh1_gap_error.append(0)
                    veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                    veh2_gap_error.append(veh2Previous_Gap-1)
                    veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                    veh3_gap_error.append(veh3Previous_Gap-1)
                    veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                    veh4_gap_error.append(veh4Previous_Gap-1)

                    totalTimeLoss.append(sum(timeLoss))
                else:
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
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
                    if ind == "5":
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
        elif options.CACC:
            if 30 < step < control_takeover_start_time + 1:
                for ind in traci.vehicle.getIDList():
                    if ind == "5":
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
                veh1_lane = traci.vehicle.getLaneID("1")
                veh2_lane = traci.vehicle.getLaneID("2")
                veh3_lane = traci.vehicle.getLaneID("3")
                veh4_lane = traci.vehicle.getLaneID("4")
                veh1_lane_value = int(veh1_lane[6])
                veh2_lane_value = int(veh2_lane[6])
                veh3_lane_value = int(veh3_lane[6])
                veh4_lane_value = int(veh4_lane[6])
                if veh1_lane_value == 1:
                    veh1_lane_state = True
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
                            traci.vehicle.setSpeed("5", 31)
                    veh1_gap_error.append(0)
                    veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                    veh2_gap_error.append(veh2Previous_Gap-1)
                    veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                    veh3_gap_error.append(veh3Previous_Gap-1)
                    veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                    veh4_gap_error.append(veh4Previous_Gap-1)

                    totalTimeLoss.append(sum(timeLoss))
                elif veh2_lane_value == 1:
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
                            traci.vehicle.setSpeed("5", 31)
                    veh1Previous_Gap = (vehPosition[0][0] - 5 - vehPosition[1][0]) / vehSpeed[1]
                    veh1_gap_error.append(veh1Previous_Gap-1)
                    veh2_gap_error.append(0)
                    veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                    veh3_gap_error.append(veh3Previous_Gap-1)
                    veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                    veh4_gap_error.append(veh4Previous_Gap-1)

                    totalTimeLoss.append(sum(timeLoss))
                elif veh1_lane_state:
                    veh1_lane_state = True
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
                            traci.vehicle.setSpeed("5", 31)
                    veh1_gap_error.append(0)
                    veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                    veh2_gap_error.append(veh2Previous_Gap-1)
                    veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                    veh3_gap_error.append(veh3Previous_Gap-1)
                    veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                    veh4_gap_error.append(veh4Previous_Gap-1)

                    totalTimeLoss.append(sum(timeLoss))
                else:
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
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
            elif control_takeover_start_time < step < end_time:
                traci.vehicle.setType("1", "Car04")
                traci.vehicle.setType("2", "Car04")
                traci.vehicle.setType("3", "Car04")
                traci.vehicle.setType("4", "Car04")
                for ind in traci.vehicle.getIDList():
                    if ind == "5":
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
                veh1_lane = traci.vehicle.getLaneID("1")
                veh2_lane = traci.vehicle.getLaneID("2")
                veh3_lane = traci.vehicle.getLaneID("3")
                veh4_lane = traci.vehicle.getLaneID("4")
                veh1_lane_value = int(veh1_lane[6])
                veh2_lane_value = int(veh2_lane[6])
                veh3_lane_value = int(veh3_lane[6])
                veh4_lane_value = int(veh4_lane[6])
                if veh1_lane_value == 1:
                    veh1_lane_state = True
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
                            traci.vehicle.setSpeed("5", 31)
                    veh1_gap_error.append(0)
                    veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                    veh2_gap_error.append(veh2Previous_Gap-1)
                    veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                    veh3_gap_error.append(veh3Previous_Gap-1)
                    veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                    veh4_gap_error.append(veh4Previous_Gap-1)

                    totalTimeLoss.append(sum(timeLoss))
                elif veh2_lane_value == 1:
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
                            traci.vehicle.setSpeed("5", 31)
                    veh1Previous_Gap = (vehPosition[0][0] - 5 - vehPosition[1][0]) / vehSpeed[1]
                    veh1_gap_error.append(veh1Previous_Gap-1)
                    veh2_gap_error.append(0)
                    veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                    veh3_gap_error.append(veh3Previous_Gap-1)
                    veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                    veh4_gap_error.append(veh4Previous_Gap-1)

                    totalTimeLoss.append(sum(timeLoss))
                elif veh1_lane_state:
                    veh1_lane_state = True
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
                            traci.vehicle.setSpeed("5", 31)
                    veh1_gap_error.append(0)
                    veh2Previous_Gap = (vehPosition[1][0] - 5 - vehPosition[2][0]) / vehSpeed[2]
                    veh2_gap_error.append(veh2Previous_Gap-1)
                    veh3Previous_Gap = (vehPosition[2][0] - 5 - vehPosition[3][0]) / vehSpeed[3]
                    veh3_gap_error.append(veh3Previous_Gap-1)
                    veh4Previous_Gap = (vehPosition[3][0] - 5 - vehPosition[4][0]) / vehSpeed[4]
                    veh4_gap_error.append(veh4Previous_Gap-1)

                    totalTimeLoss.append(sum(timeLoss))
                else:
                    for ind in traci.vehicle.getIDList():
                        if ind == "5":
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
                laneChangeDecision.append(0)

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
                    laneChangeDecision.append(0)
                    platoon_lane_ids = []
                    for x in [1, 2, 3, 4]:
                        platoon_lane_ids.append(traci.vehicle.getLaneID(f"{x}"))
                    veh1Speed = traci.vehicle.getSpeed("1")
                    veh2Speed = traci.vehicle.getSpeed("2")
                    veh3Speed = traci.vehicle.getSpeed("3")
                    veh4Speed = traci.vehicle.getSpeed("4")
                    if check_Lane_Change(platoon_lane_ids):
                        traci.vehicle.changeLane("1", 1, 300)
                        traci.vehicle.changeLane("2", 1, 300)
                        traci.vehicle.changeLane("3", 1, 300)
                        traci.vehicle.changeLane("4", 1, 300)

                        avgTimeLoss = sum(timeLoss)/4
                        timeLossChangeRate = sum([a - b for a, b in zip(timeLoss, previousTimeLoss)])/4

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

                        # timeLoss.append(traci.vehicle.getTimeLoss("1"))
                        timeLossChangeRate = sum([a - b for a, b in zip(timeLoss, previousTimeLoss)])/4
                        previousTimeLoss = timeLoss
                        traci.vehicle.setSpeed("1", veh1Speed)

                        time_to_collision[0] = 0
                        TTL = np.vstack([TTL, np.array(time_to_collision)])
                    else:
                        veh_id_to_slow_down = []
                        veh_id_to_speed_up = []
                        for ind, x in enumerate(platoon_lane_ids):
                            veh_id = ind + 1
                            if x != 1:
                                if veh_id > 3:
                                    veh_id_to_slow_down.append(veh_id)
                                    veh_id_to_speed_up.append(veh_id - 1)
                                else:
                                    veh_id_to_slow_down.append(veh_id + 1)
                                traci.vehicle.changeLane(f"{veh_id}", 1, 300)
                            else:
                                pass

                        for x in veh_id_to_slow_down:
                            veh_speed_to_slow_down = traci.vehicle.getSpeed(f"{x}")
                            traci.vehicle.setSpeed(f"{x}", veh_speed_to_slow_down - 0.5)

                        for x in veh_id_to_speed_up:
                            veh_speed_to_speed_up = traci.vehicle.getSpeed(f"{x}")
                            traci.vehicle.setSpeed(f"{x}", veh_speed_to_speed_up + 0.5)

                        avgTimeLoss = sum(timeLoss)/4
                        timeLossChangeRate = sum([a - b for a, b in zip(timeLoss, previousTimeLoss)])/4

                        veh2 = fuzzyLogic.calc_Inputs(2, vehPosition[1][0], veh2Previous_Gap, vehPosition[2][0], vehSpeed[2], vehicleGapErrors, avgTimeLoss, timeLossChangeRate, SUMO, SUMOLANECHANGE)
                        veh2Previous_Gap = veh2[0]
                        veh2_gap.append(veh2[0])
                        vehicleGapErrors.append(veh2[1])
                        veh2Acceleration = veh2[3]
                        veh2Speed = vehSpeed[2] + veh2Acceleration
                        veh2_gap_error.append(veh2[1])
                        veh2_gap_error_rate.append(veh2[2])
                        veh2_lane_change_decision.append(veh2[4])

                        veh3 = fuzzyLogic.calc_Inputs(3, vehPosition[2][0], veh3Previous_Gap, vehPosition[3][0], vehSpeed[3], vehicleGapErrors, avgTimeLoss, timeLossChangeRate, SUMO, SUMOLANECHANGE)
                        veh3Previous_Gap = veh3[0]
                        veh3_gap.append(veh3[0])
                        vehicleGapErrors.append(veh3[1])
                        veh3Acceleration = veh3[3]
                        veh3Speed = vehSpeed[3] + veh3Acceleration
                        veh3_gap_error.append(veh3[1])
                        veh3_gap_error_rate.append(veh3[2])
                        veh3_lane_change_decision.append(veh3[4])

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

                        # timeLoss.append(traci.vehicle.getTimeLoss("1"))
                        timeLossChangeRate = sum([a - b for a, b in zip(timeLoss, previousTimeLoss)])/4
                        previousTimeLoss = timeLoss

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

                    # Lane availability
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
                                laneChangeDecision.append(1)
                                # print(traci.vehicle.getLaneID("1"))
                            else:
                                laneChangeDecision.append(0)
                        else:
                            laneChangeDecision.append(0)
                    else:
                        laneChangeDecision.append(0)

                    timeLossChangeRate = sum([a - b for a, b in zip(timeLoss, previousTimeLoss)])/4
                    previousTimeLoss = timeLoss
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
    return veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error, veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, TTL, totalTimeLoss, laneChangeDecision


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

    if options.highway_2_mod:
        highway_filename = "highway_2_mod"
    else:
        highway_filename = "highway_2"

    fileName_no_sffix = f"{fileName}_{highway_filename}"
    control_takeover_start_time = 300
    end_time = 2000

    timestr = time.strftime("%Y%m%d_%H%M%S")

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
        veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, \
        TTL, totalTimeLoss, laneChangeDecision = run(fis_start_time, end_time)

    veh1_fitness_sum = sum(abs(veh1_gap_error[fis_start_time:end_time]))
    veh2_fitness_sum = sum(abs(veh2_gap_error[fis_start_time:end_time]))
    veh3_fitness_sum = sum(abs(veh3_gap_error[fis_start_time:end_time]))
    veh4_fitness_sum = sum(abs(veh4_gap_error[fis_start_time:end_time]))

    fitness = np.sum(abs([veh1_fitness_sum, veh2_fitness_sum, veh3_fitness_sum, veh4_fitness_sum]))

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
    elif options.CACC:
        title = "CACC"
    elif options.GFS:
        title = "GFS"
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

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(laneChangeDecision)
    posFile = f'./{images_subdirectory}/{title}_vehicle_laneChangeDecision.png'
    if os.path.isfile(posFile):
        os.unlink(posFile)
    fig.savefig(f'{posFile}')
