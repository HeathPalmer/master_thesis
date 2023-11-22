#!/usr/bin/env python
import csv
from EasyGA import GA, crossover, mutation
import matplotlib.pyplot as plt
import ntpath
import numpy as np
import os
import optparse
import random
import sys
import time
import traci
from sumolib import checkBinary  # Checks for the binary in environ vars
from test_fuzzy_controller_live import FuzzyHWClass
# import xml.etree.ElementTree as ET
# necessary to import xml2csv file from a different directory
# source:https://www.codegrepper.com/code-examples/python/import+script+from+another+folder+python
sys.path.append('C:/Program Files (x86)/Eclipse/Sumo/tools/xml')

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

    def initialize_population(self):
        """Custom population initialization with chromosomes
        """
        # ga_test = My_GA()

        self.population = self.make_population(
            create_pop()
            for _
            in range(self.population_size)
        )

        #self.population[0] = [-1, 0, 15, 14, 22, 29, 25, 50, 100, -0.1, 0, 0.03, 0.025, 0.05, 0.07, 0.065, 0.1, 1, -0.1, 0, 0.49, 0.48, 1, 1.1]
        # print(self.population[0])
        # print(f"The population is: {self.population}")

    # def crossover_individual_impl(self, parent_1, parent_2, weight=0.5):
    #     """Cross two parents by swapping genes at one random point."""

    #     minimum_parent_length = min(len(parent_1), len(parent_2))

    #     # Weighted random integer from 0 to minimum parent length - 1
    #     swap_index = int(ga.weighted_random(weight) * minimum_parent_length)

    #     ga.population.add_child(parent_1[:swap_index] + parent_2[swap_index:])
    #     ga.population.add_child(parent_2[:swap_index] + parent_1[swap_index:])


def create_pop():
    membershipFunction = []
    for x in range(3):
        outterBounds = [random.uniform(-1, 100), random.uniform(-1, 100)]
        outterBounds.sort()
        peak = random.uniform(outterBounds[0], outterBounds[1])
        membershipFunction.append(outterBounds[0])
        membershipFunction.append(peak)
        membershipFunction.append(outterBounds[1])

    for x in range(3):
        outterBounds = [random.uniform(-0.1, 1), random.uniform(-0.1, 1)]
        outterBounds.sort()
        peak = random.uniform(outterBounds[0], outterBounds[1])
        membershipFunction.append(outterBounds[0])
        membershipFunction.append(peak)
        membershipFunction.append(outterBounds[1])

    for x in range(2):
        outterBounds = [random.uniform(-0.1, 1.1), random.uniform(-0.1, 1.1)]
        outterBounds.sort()
        peak = random.uniform(outterBounds[0], outterBounds[1])
        membershipFunction.append(outterBounds[0])
        membershipFunction.append(peak)
        membershipFunction.append(outterBounds[1])

    return membershipFunction


# fitness function
def user_def_fitness(chromosome):
    try:
        global ERROR_OCCURED
        ERROR_OCCURED = False

        # check if another fitness is being ran
        # if traci.simulation.getDeltaT() > 0:
        #     time.sleep(600)
        # else:
        #     pass
        # ERROR_OCCURED = False
        # print(f"The solution was: {chromosome}")
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
        # print(chromosome_array_of_arrays)
        # print(f"The proposed chromosome is {chromosome_array_of_arrays}")

        # starting SUMO and traci

        global fullOutFileName, fcdOutInfoFileName, recnum
        # set the file name based on increamenting value
        # i = 0
        # while os.path.exists(os.path.join(spreadsheet_subdirectory, "%s_fcdout.xml" % format(int(i), '03d'))):
        #     i += 1
        # recnum = format(int(i), '03d')
        recnum = 100
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
                     "--start",
                     "--quit-on-end"
                     ])
        # removed: "--full-output", fullOutFileName,
        #  "--fcd-output", fcdOutInfoFileName,
        #  "--fcd-output.acceleration"
        time.sleep(1)

        global veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error, \
            veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, TTL, totalTimeLoss

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
            veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, TTL, totalTimeLoss = run(fis_start_time, end_time, chromosome_array_of_arrays)

        veh1_fitness_sum = sum(veh1_gap_error[fis_start_time:end_time])
        veh2_fitness_sum = sum(veh2_gap_error[fis_start_time:end_time])
        veh3_fitness_sum = sum(veh3_gap_error[fis_start_time:end_time])
        veh4_fitness_sum = sum(veh4_gap_error[fis_start_time:end_time])

        fitness = np.sum([veh1_fitness_sum, veh2_fitness_sum, veh3_fitness_sum, veh4_fitness_sum])
        # fitness = sum(totalTimeLoss)

    except Exception as e:
        # fitness = np.sum([veh1_fitness_sum, veh2_fitness_sum, veh3_fitness_sum, veh4_fitness_sum])
        ERROR_OCCURED = True
        print(f"There was an error calculating the fitness. The error was: {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    finally:
        # traci.close()
        # sys.stdout.flush()
        if ERROR_OCCURED is True:
            fitness = 100000
        else:
            print(f"No error occured. The fitness is: {fitness}")
        print(f"Attempted to run fitness with a fitness of: {fitness}")
        # del veh1_fitness_sum, veh2_fitness_sum, veh3_fitness_sum, veh4_fitness_sum, \
        #     veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate
        # # print("The garbace count is: ", gc.get_count())
        # gc.collect()
        return fitness


# Custom implementation of the survivor selection method.
def survivor_selection_impl(ga):
    needed_amount = len(ga.population) - len(ga.population.next_population)
    # print(f"population: {ga.population}")
    # print(f"population length is: {len(ga.population)}")
    # print(f"The next population length is: {len(ga.population.next_population)}")
    print(f"needed amount: {needed_amount}")
    # print(f"next generation is: {ga.population.next_population}")

    best_chromosome = ga.population[0]
    # Save the best chromosome
    if needed_amount > 0:
        ga.population.add_child(best_chromosome)
        needed_amount -= 1
        # print(f"population: {ga.population}")
        print(f"needed amount: {needed_amount}")
        # print(f"next generation is: {ga.population.next_population}")
        if needed_amount <= 0:
            return

    # Loop through the population
    for chromosome in ga.population:
        print(f"the chomosome fitness is: {chromosome.fitness}")
        if chromosome.fitness == 100000:
            new_chrom = create_pop()
            print(f"new chrom is: {new_chrom}")
            ga.population.add_child(new_chrom)
            needed_amount -= 1
            if needed_amount <= 0:
                break
        # Add chromosome if any of the genes are different
        if any(best_gene != gene for best_gene, gene in zip(best_chromosome, chromosome)):
            ga.population.add_child(chromosome)
            needed_amount -= 1
            # print(f"population: {ga.population}")
            print(f"needed amount: {needed_amount}")
            # print(f"next generation is: {ga.population.next_population}")

            # Stop if enough chromosomes survive
            if needed_amount <= 0:
                break
    print(f"population length is: {len(ga.population)}")
    print(f"The next population length is: {len(ga.population.next_population)}")


# Custom implementation of the mutation population method.
# @GA._check_chromosome_mutation_rate
# @GA._loop_selections
# def mutation_population_impl(ga):
#     print("Mutating...")
#     for i in range(len(ga.population)):
#         print(ga.population[i].fitness)
#         if ga.population[i].fitness == 100000:
#             new_chromosome = create_pop()
#             for j in range(66):
#                 ga.population[i][j] = new_chromosome[j]
#             # recalculate the fitness value. Is this required?
#             # ga.population[i].fitness = user_def_fitness(ga.population[i])
#             print("Mutated the whole chromosome.")
#         else:
#             pass
#         i += 1
#     actual_mutation_rate = ga.gene_mutation_rate * 0.05
#     random_chance = random.randint(0, 100)
#     mutation_chance = 100 * actual_mutation_rate
#     if random_chance < mutation_chance:
#         # indexes surrounding the middle of the population
#         low_index = int(len(ga.population)*(1-actual_mutation_rate)/2)
#         # upper index is the last individual
#         upp_index = int(len(ga.population))

#         index = random.randrange(low_index, upp_index)
#         ga.mutation_individual_impl(ga.population[index])
#         print("Mutated genes in the chromosome")
#     else:
#         print("skipped gene mutation")
#         pass


# Custom implementation of the mutation individual method.
# @mutation.Mutation._check_gene_mutation_rate
# @mutation.Mutation._loop_mutations
# def mutation_individual_impl(ga, chromosome):

#     index = random.randrange(0, len(chromosome))

#     # Use swapping
#     if index > len(chromosome)/2:
#         index_1 = index
#         index_2 = random.randrange(index, len(chromosome))  # Stay in range.

#         chromosome[index_1], chromosome[index_2] = chromosome[index_2], chromosome[index_1]  # Swap genes.

#     # Make a new gene using chromosome_impl
#     elif ga.chromosome_impl is not None:
#         chromosome[index] = ga.chromosome_impl()[index]

#     # Make a new gene using gene impl
#     elif ga.gene_impl is not None:
#         chromosome[index] = ga.gene_impl()

#     # Can't make new gene
#     else:
#         raise Exception("Did not specify any initialization constraints.")


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
def run(fis_start_time, end_time, chromosome_array_of_arrays):
    global ERROR_OCCURED
    ERROR_OCCURED = False
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

        veh1_lane_change_decision = []
        veh2_lane_change_decision = []
        veh3_lane_change_decision = []
        veh4_lane_change_decision = []

        totalTimeLoss = []

        TTL = np.empty((0, 4), int)
        # apparently this set of membership functions does not work...
        GA_chromosome = [[-1.4457460055267017],[0.9596282534579326],[7],[-1.9687233494625112],[-1.4949227177070799],[0.4612586650427106],[-0.5184262059508764],[0.33112194921412064],[1],[0.032862387192658105],[1.1411546493663296],[1.6209736373574386],[-1.325107367543487],[-1.0780860231020188],[-0.5188892088706325],[0.35693070676711347],[2.050684365001913],[2.774034566185253],[1],[2.68280418027796],[2.748702738024731],[2],[3.489405091498025],[5.0337841895636455],[-5.724586988002178],[-0.35251959938796507],[1],[-4.750454037624641],[-4.254990848572678],[7],[1],[3.2731361566159016],[7],[-5.90119953417147],[-3.844035790352814],[3.8775225627050975],[-5.2362274660785495],[-1.168961150316191],[9],[0.23788695355013978],[3.1587191623626643],[4.687913542495231],[1],[7.525508667366361],[7.804668032310376],[1.342919268417213],[1.39218145634533],[1.6151422656498324],[0.31956898205956374],[1.0963867260814184],[2.2544998100268643],[1],[2.490873025143525],[4.1265360934946305],[-0.6069999197682163],[0.6492415564704332],[1],[-0.1372509781448512],[1.347515967623831],[1.3896826810635972],[-3.6379127234475956],[-1.6691021049343568],[0.2178198719024973],[-4.54308179374002],[0.0989252235612108],[1.3340108981072287]]

        i = 0
        new_data = []
        while i < len(GA_chromosome):
            new_data.append(GA_chromosome[i])
            i = i+1

        j = 0
        GA_chromosome = []
        while j < len(new_data):
            GA_chromosome.append(new_data[j:j+3])
            j = j + 3

        # convert array of arrays to a Numpy array of arrays
        GA_chromosome = np.array(GA_chromosome)

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
        SUMOLANECHANGE = fuzzyLogic.createFuzzyLaneControl(chromosome_array_of_arrays)
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

                    traci.vehicle.setSpeed("5", 31)
                    veh1_lane = traci.vehicle.getLaneID("1")
                    veh1_lane_value = int(veh1_lane[6])
                    if veh1_lane_value == 1:
                        # print(veh1_lane_value)
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
                        avgTimeLoss = sum(timeLoss)/4
                        timeLoss.append(traci.vehicle.getTimeLoss("0"))
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
    except Exception as e:
        ERROR_OCCURED = True
        print(f"Error in the FIS run function at step {step}. Error was {e} \n")
        print(f"The state of error is {ERROR_OCCURED}")

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    finally:
        traci.close()
        sys.stdout.flush()
        time.sleep(1)
        if ERROR_OCCURED is True:
            step_difference = 1064 - step
            i = 0
            for i in range(step_difference):
                veh1_gap_error.append(10)
                veh2_gap_error.append(10)
                veh3_gap_error.append(10)
                veh4_gap_error.append(10)

                veh1_gap_error_rate.append(10)
                veh2_gap_error_rate.append(10)
                veh3_gap_error_rate.append(10)
                veh4_gap_error_rate.append(10)

                i = i+1
        else:
            print("No error in the FIS run function")
        return veh1_gap_error, veh2_gap_error, veh3_gap_error, veh4_gap_error, \
            veh1_gap_error_rate, veh2_gap_error_rate, veh3_gap_error_rate, veh4_gap_error_rate, TTL, totalTimeLoss


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
    global ERROR_OCCURED
    ERROR_OCCURED = False
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

    fileName_No_Suffix = "highway_2"
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

    ga.generation_goal = 100
    ga.chromosome_length = 66
    ga.population_size = 10
    # Probability that the best selected chromosome will be selected as a parent
    ga.selection_probability = 0.5
    # ga.fitness_goal = 0
    # Run everything.
    ga.crossover_individual_impl = crossover.Crossover.Individual.single_point
    ga.survivor_selection_impl = survivor_selection_impl
    # create custom mutation method
    #
    ga.fitness_function_impl = user_def_fitness
    ga.target_fitness_type = 'min'
    # This makes it so the mutation functions are called every time
    # This does not mean that the chromosome will be mutated every time.
    # This is to ensure that chromosomes out of bounds are replaced.
    ga.chromosome_mutation_rate = 0.3  # Rate at which chromosomes get mutated
    # Ratio of chromosomes selected (1.0 = 1 parent for every chromosome in the population)
    ga.parent_ratio = 0.1  # this should make it so that only two parents are chosen to mate if the population is 10
    # ga.gene_mutation_rate = 1
    ga.adapt_probability_rate = 0.1
    # If you don't want to store all data coming from the GA set to
    # false. This will also relieve some memory density issues.
    ga.save_data = True
    ga.database_name = f'{spreadsheet_subdirectory}/database.db'
    if os.path.isfile(ga.database_name):
        os.unlink(ga.database_name)

    generation_information_file = f'{spreadsheet_subdirectory}/ga_information.csv'
    if os.path.isfile(generation_information_file):
        os.unlink(generation_information_file)
    # Setting the mutation methods.
    # ga.mutation_population_impl = mutation_population_impl
    # ga.mutation_individual_impl = mutation_individual_impl

    # record when GA begins to train
    training_start_time = time.time()

    while ga.active():
        # Evolve only a certain number of generations
        ga.evolve(1)

        ga.print_population()
        # Print the current generation
        ga.print_generation()
        # Print the best chromosome from that generations population
        ga.print_best_chromosome()
        # If you want to show each population
        # ga.print_population()
        # To divide the print to make it easier to look at
        print('-'*75)

        # write results to CSV instead of the SQL database.
        generation_info = []
        current_gen = ga.current_generation
        current_best_fitness = ga.population[0].fitness
        currenet_best_chromosome = ga.population[0]
        current_population_fitness = []
        for i in range(len(ga.population)):
            current_population_fitness.append(ga.population[i].fitness)
        current_population = ga.population

        training_generation_end_time = time.time()
        total_training_time = training_generation_end_time - training_start_time

        generation_info = [current_gen, total_training_time, current_best_fitness, currenet_best_chromosome, current_population_fitness, current_population]

        with open(generation_information_file, 'a') as outfile:
            training_end_time = time.time()
            writer = csv.writer(outfile)
            writer.writerow(generation_info)

        # for i in range(len(ga.population)):
        #     print(ga.population[i].fitness)
        #     if ga.population[i].fitness == 100000:
        #         new_chromosome = create_pop()
        #         for j in range(66):
        #             ga.population[i][j] = new_chromosome[j]
        #         # ga.population[i].fitness = user_def_fitness(ga.population[i])
        #     else:
        #         pass
        #     i += 1
    # print(ga.population)

    time.sleep(1)

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"The GFS took {total_training_time} seconds to train")

    time.sleep(2)

    # print(ga.database.generation_total_fitness("average"))
    ga.graph.lowest_value_chromosome()
    ga.graph.show()
