from matplotlib import pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd


class FuzzyHWClass:
    def createFuzzyControl(self, membership_function_values):

        # initialize fuzy variables
        self.gap_error = ctrl.Antecedent(np.arange(-2, 3, 0.01), 'gap-error-value')
        self.gap_error_rate = ctrl.Antecedent(np.arange(-1, 1, 0.001), 'gap-error-change-rate-value')
        # self.gap_diff_from_min = ctrl.Antecedent(np.arange(-3, 3, 0.01), 'gap-diff-from-avg')

        # output acceleration
        self.acceleration = ctrl.Consequent(np.arange(-5, 5.1, 0.001), 'acceleration-value')

        # Function for fuzz.trimf(input,left edge, center edge, right edge)
        self.gap_error['ExtraExtraExtraSmall'] = fuzz.trimf(self.gap_error.universe, membership_function_values[0])
        self.gap_error['ExtraExtraSmall'] = fuzz.trimf(self.gap_error.universe, membership_function_values[1])
        self.gap_error['ExtraSmall'] = fuzz.trimf(self.gap_error.universe, membership_function_values[2])
        self.gap_error['Small'] = fuzz.trimf(self.gap_error.universe, membership_function_values[3])
        self.gap_error['Medium'] = fuzz.trimf(self.gap_error.universe, membership_function_values[4])
        self.gap_error['Large'] = fuzz.trimf(self.gap_error.universe, membership_function_values[5])
        self.gap_error['ExtraLarge'] = fuzz.trimf(self.gap_error.universe, membership_function_values[6])
        # self.gap_error.view()
        # input("Press Enter to continue...")

        self.gap_error_rate['ExtraExtraExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[7])
        self.gap_error_rate['ExtraExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[8])
        self.gap_error_rate['ExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[9])
        self.gap_error_rate['Small'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[10])
        self.gap_error_rate['Medium'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[11])
        self.gap_error_rate['Large'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[12])
        self.gap_error_rate['ExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[13])
        self.gap_error_rate['ExtraExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[14])
        # self.gap_error_rate.view()
        # input("Press Enter to continue...")

        # self.gap_diff_from_min['Small'] = fuzz.trimf(self.gap_diff_from_min.universe, [-3, -1.5, 0])
        # self.gap_diff_from_min['Medium'] = fuzz.trimf(self.gap_diff_from_min.universe, [-1.5, 0, 1.5])
        # self.gap_diff_from_min['Large'] = fuzz.trimf(self.gap_diff_from_min.universe, [0, 1.5, 2])

        # setup the 12 output membership functions
        self.acceleration['ExtraExtraSmall'] = fuzz.trimf(self.acceleration.universe, membership_function_values[15])
        self.acceleration['ExtraSmall'] = fuzz.trimf(self.acceleration.universe, membership_function_values[16])
        self.acceleration['Small'] = fuzz.trimf(self.acceleration.universe, membership_function_values[17])
        self.acceleration['Medium'] = fuzz.trimf(self.acceleration.universe, membership_function_values[18])
        self.acceleration['Large'] = fuzz.trimf(self.acceleration.universe, membership_function_values[19])
        self.acceleration['ExtraLarge'] = fuzz.trimf(self.acceleration.universe, membership_function_values[20])
        self.acceleration['ExtraExtraLarge'] = fuzz.trimf(self.acceleration.universe, membership_function_values[21])
        # self.acceleration.view()
        # input("Press Enter to continue...")

        # STAGE TWO: DEFINE RULE BASE AND INFERENCE USING SCALED OUTPUT APPROACH
        rule1 = ctrl.Rule(antecedent=((self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraExtraSmall']) |
                                      (self.gap_error['ExtraExtraExtraSmall'] & self.gap_error_rate['ExtraExtraExtraSmall'])),
                          consequent=self.acceleration['ExtraExtraSmall'])

        rule2 = ctrl.Rule(antecedent=(self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraExtraSmall']),
                          consequent=self.acceleration['ExtraSmall'])

        rule3 = ctrl.Rule(antecedent=((self.gap_error['Small'] & self.gap_error_rate['ExtraExtraSmall']) |
                                      (self.gap_error['Medium'] & self.gap_error_rate['ExtraExtraSmall']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['ExtraExtraSmall']) |
                                      (self.gap_error['ExtraLarge'] & self.gap_error_rate['ExtraExtraSmall']) |
                                      (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraSmall']) |
                                      (self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraSmall']) |
                                      (self.gap_error['Small'] & self.gap_error_rate['ExtraSmall']) |
                                      (self.gap_error['Medium'] & self.gap_error_rate['ExtraSmall']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['ExtraSmall']) |
                                      (self.gap_error['ExtraLarge'] & self.gap_error_rate['ExtraSmall']) |
                                      (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['Small']) |
                                      (self.gap_error['ExtraSmall'] & self.gap_error_rate['Small']) |
                                      (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['Medium']) |
                                      (self.gap_error['ExtraSmall'] & self.gap_error_rate['Medium'])),
                          consequent=self.acceleration['Small'])

        rule4 = ctrl.Rule(antecedent=((self.gap_error['Small'] & self.gap_error_rate['Small']) |
                                      (self.gap_error['Medium'] & self.gap_error_rate['Small']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['Small']) |
                                      (self.gap_error['ExtraLarge'] & self.gap_error_rate['Small']) |
                                      (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['Large']) |
                                      (self.gap_error['ExtraSmall'] & self.gap_error_rate['Large']) |
                                      (self.gap_error['Medium'] & self.gap_error_rate['Medium'])),
                          consequent=self.acceleration['Medium'])

        rule5 = ctrl.Rule(antecedent=((self.gap_error['Small'] & self.gap_error_rate['Medium']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['Medium']) |  # added this here
                                      (self.gap_error['ExtraLarge'] & self.gap_error_rate['Medium']) |
                                      (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraLarge']) |
                                      (self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraLarge']) |
                                      (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraExtraLarge']) |
                                      (self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraExtraLarge'])),
                          consequent=self.acceleration['Large'])

        rule6 = ctrl.Rule(antecedent=((self.gap_error['Medium'] & self.gap_error_rate['Large']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['Large']) |
                                      (self.gap_error['Small'] & self.gap_error_rate['ExtraLarge']) |
                                      (self.gap_error['Medium'] & self.gap_error_rate['ExtraLarge'])),
                          consequent=self.acceleration['ExtraLarge'])

        rule7 = ctrl.Rule(antecedent=((self.gap_error['ExtraLarge'] & self.gap_error_rate['Large']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['ExtraLarge']) |
                                      (self.gap_error['ExtraLarge'] & self.gap_error_rate['ExtraLarge']) |
                                      (self.gap_error['Small'] & self.gap_error_rate['ExtraExtraLarge']) |
                                      (self.gap_error['Medium'] & self.gap_error_rate['ExtraExtraLarge']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['ExtraExtraLarge']) |
                                      (self.gap_error['ExtraLarge'] & self.gap_error_rate['ExtraExtraLarge'])),
                          consequent=self.acceleration['ExtraExtraLarge'])

        # rule1.view()

        SUMO_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])

        SUMO = ctrl.ControlSystemSimulation(SUMO_control, flush_after_run=1000)  # using cache=False did not help
        return SUMO

    def createFuzzyLaneControl(self, membership_function_values):

        # initialize fuzy variables
        self.avg_time_loss = ctrl.Antecedent(np.arange(0, 100, 1), 'avg-time-loss')
        self.avg_time_loss_rate = ctrl.Antecedent(np.arange(-2, 1, 0.005), 'avg-time-loss-rate')
        # self.gap_diff_from_min = ctrl.Antecedent(np.arange(-3, 3, 0.01), 'gap-diff-from-avg')

        # output acceleration
        self.change_lane_decision = ctrl.Consequent(np.arange(-1, 2, 0.01), 'change-lane-decision')

        # Function for fuzz.trimf(input,left edge, center edge, right edge)
        self.avg_time_loss['Small'] = fuzz.trimf(self.avg_time_loss.universe, membership_function_values[0])
        self.avg_time_loss['Medium'] = fuzz.trimf(self.avg_time_loss.universe, membership_function_values[1])
        self.avg_time_loss['Large'] = fuzz.trimf(self.avg_time_loss.universe, membership_function_values[2])
        # self.avg_time_loss.view()
        # input("Press Enter to continue...")

        self.avg_time_loss_rate['Small'] = fuzz.trimf(self.avg_time_loss_rate.universe, membership_function_values[3])
        self.avg_time_loss_rate['Medium'] = fuzz.trimf(self.avg_time_loss_rate.universe, membership_function_values[4])
        self.avg_time_loss_rate['Large'] = fuzz.trimf(self.avg_time_loss_rate.universe, membership_function_values[5])
        # self.avg_time_loss_rate.view()
        # input("Press Enter to continue...")

        # setup the 12 output membership functions
        self.change_lane_decision['NoChange'] = fuzz.trimf(self.change_lane_decision.universe, membership_function_values[6])
        self.change_lane_decision['ChangeLane'] = fuzz.trimf(self.change_lane_decision.universe, membership_function_values[7])
        # self.change_lane_decision.view()
        # input("Press Enter to continue...")

        # STAGE TWO: DEFINE RULE BASE AND INFERENCE USING SCALED OUTPUT APPROACH
        rule1 = ctrl.Rule(antecedent=((self.avg_time_loss['Small'] & self.avg_time_loss_rate['Small']) |
                                      (self.avg_time_loss['Small'] & self.avg_time_loss_rate['Medium']) |
                                      (self.avg_time_loss['Medium'] & self.avg_time_loss_rate['Small'])),
                          consequent=self.change_lane_decision['NoChange'])

        rule2 = ctrl.Rule(antecedent=((self.avg_time_loss['Small'] & self.avg_time_loss_rate['Large']) |
                                      (self.avg_time_loss['Medium'] & self.avg_time_loss_rate['Medium']) |
                                      (self.avg_time_loss['Medium'] & self.avg_time_loss_rate['Large']) |
                                      (self.avg_time_loss['Large'] & self.avg_time_loss_rate['Small']) |
                                      (self.avg_time_loss['Large'] & self.avg_time_loss_rate['Medium']) |
                                      (self.avg_time_loss['Large'] & self.avg_time_loss_rate['Large'])),
                          consequent=self.change_lane_decision['ChangeLane'])

        # rule1.view()

        SUMO_control = ctrl.ControlSystem([rule1, rule2])

        SUMO = ctrl.ControlSystemSimulation(SUMO_control, flush_after_run=1000)  # using cache=False did not help
        return SUMO

    def createSecondFuzzyLongitudinalControl(self, membership_function_values):

        # initialize fuzy variables
        self.platoon_gap_error = ctrl.Antecedent(np.arange(-10, 10, 0.01), 'platoon-gap-error-value')
        self.velocity_error = ctrl.Antecedent(np.arange(-31.292, 3, 0.01), 'vehicle-error-value')
        # self.gap_diff_from_min = ctrl.Antecedent(np.arange(-3, 3, 0.01), 'gap-diff-from-avg')

        # output acceleration
        self.acceleration = ctrl.Consequent(np.arange(-5, 5.1, 0.01), 'acceleration-value')

        # Function for fuzz.trimf(input,left edge, center edge, right edge)
        self.platoon_gap_error['Small'] = fuzz.trimf(self.platoon_gap_error.universe, membership_function_values[0])
        self.platoon_gap_error['Medium'] = fuzz.trimf(self.platoon_gap_error.universe, membership_function_values[1])
        self.platoon_gap_error['Large'] = fuzz.trimf(self.platoon_gap_error.universe, membership_function_values[2])
        # self.platoon_gap_error.view()
        # input("Press Enter to continue...")

        self.velocity_error['Small'] = fuzz.trimf(self.velocity_error.universe, membership_function_values[3])
        self.velocity_error['Medium'] = fuzz.trimf(self.velocity_error.universe, membership_function_values[4])
        self.velocity_error['Large'] = fuzz.trimf(self.velocity_error.universe, membership_function_values[5])
        # self.velocity_error.view()
        # input("Press Enter to continue...")
        # self.gap_diff_from_min['Small'] = fuzz.trimf(self.gap_diff_from_min.universe, [-3, -1.5, 0])
        # self.gap_diff_from_min['Medium'] = fuzz.trimf(self.gap_diff_from_min.universe, [-1.5, 0, 1.5])
        # self.gap_diff_from_min['Large'] = fuzz.trimf(self.gap_diff_from_min.universe, [0, 1.5, 2])

        # setup the 12 output membership functions
        self.acceleration['Small'] = fuzz.trimf(self.acceleration.universe, membership_function_values[6])
        self.acceleration['Medium'] = fuzz.trimf(self.acceleration.universe, membership_function_values[7])
        self.acceleration['Large'] = fuzz.trimf(self.acceleration.universe, membership_function_values[8])
        # self.acceleration.view()
        # input("Press Enter to continue...")

        # STAGE TWO: DEFINE RULE BASE AND INFERENCE USING SCALED OUTPUT APPROACH
        rule1 = ctrl.Rule(antecedent=((self.platoon_gap_error['Small'] & self.velocity_error['Small']) |
                                      (self.platoon_gap_error['Small'] & self.velocity_error['Medium']) |
                                      (self.platoon_gap_error['Medium'] & self.velocity_error['Small']) |
                                      (self.platoon_gap_error['Large'] & self.velocity_error['Small']) |
                                      (self.platoon_gap_error['Large'] & self.velocity_error['Medium'])),
                          consequent=self.acceleration['Small'])

        rule2 = ctrl.Rule(antecedent=((self.platoon_gap_error['Small'] & self.velocity_error['Large']) |
                                      (self.platoon_gap_error['Medium'] & self.velocity_error['Medium'])),
                          consequent=self.acceleration['Medium'])

        rule3 = ctrl.Rule(antecedent=((self.platoon_gap_error['Medium'] & self.velocity_error['Large']) |
                                      (self.platoon_gap_error['Large'] & self.velocity_error['Large'])),
                          consequent=self.acceleration['Large'])

        # rule1.view()

        SUMO_control = ctrl.ControlSystem([rule1, rule2, rule3])

        SUMO = ctrl.ControlSystemSimulation(SUMO_control, flush_after_run=1000)  # using cache=False did not help
        return SUMO

    # @profile
    def fuzzyHW(self, vehicle_id, vehicle_gap_error, vehicle_gap_error_rate, SUMO):  # , *ego_gap_diff_from_min):
        # inputs = [vehicle_id, vehicle_gap_error, vehicle_gap_error_rate]  # , ego_gap_diff_from_min]

        # if membership_function_values.any():
        #     pass
        # else:
        #     membership_function_values = np.array([
        #         [-2, -1, -0.5],
        #         [-0.6, -0.5, -0.25],
        #         [-0.5, -0.25, 0],
        #         [-0.25, 0, 0.25],
        #         [0, 0.5, 1],
        #         [0.5, 1, 1.5],
        #         [1, 1.5, 3],
        #         # second input mem functions
        #         [-10, -7.5, -5.6],
        #         [-6, -5.36, -2.235],
        #         [-5.36, -2.235, -0.447],
        #         [-10, -2.235, 0],
        #         [-0.447, 0, 0.447],
        #         [0, 0.447, 2.235],
        #         [0.447, 2.235, 5.36],
        #         [2.235, 5.36, 10],
        #         # output membership functions
        #         [-5, -4.572, -3],
        #         [-4.572, -3, -1.5],
        #         [-2.235, -1.5, 0],
        #         [-1.5, 0, 1.5],
        #         [0, 1.5, 3],
        #         [1.5, 3, 4.572],
        #         [3, 4.572, 5]
        #     ])

        # # initialize fuzy variables
        # self.gap_error = ctrl.Antecedent(np.arange(-2, 3, 0.01), 'gap-error-value')
        # self.gap_error_rate = ctrl.Antecedent(np.arange(-6, 10, 0.001), 'gap-error-change-rate-value')
        # # self.gap_diff_from_min = ctrl.Antecedent(np.arange(-3, 3, 0.01), 'gap-diff-from-avg')

        # # output acceleration
        # self.acceleration = ctrl.Consequent(np.arange(-5, 5.1, 0.001), 'acceleration-value')

        # # Function for fuzz.trimf(input,left edge, center edge, right edge)
        # self.gap_error['ExtraExtraExtraSmall'] = fuzz.trimf(self.gap_error.universe, membership_function_values[0])
        # self.gap_error['ExtraExtraSmall'] = fuzz.trimf(self.gap_error.universe, membership_function_values[1])
        # self.gap_error['ExtraSmall'] = fuzz.trimf(self.gap_error.universe, membership_function_values[2])
        # self.gap_error['Small'] = fuzz.trimf(self.gap_error.universe, membership_function_values[3])
        # self.gap_error['Medium'] = fuzz.trimf(self.gap_error.universe, membership_function_values[4])
        # self.gap_error['Large'] = fuzz.trimf(self.gap_error.universe, membership_function_values[5])
        # self.gap_error['ExtraLarge'] = fuzz.trimf(self.gap_error.universe, membership_function_values[6])
        # # print(self.gap_error.view())

        # self.gap_error_rate['ExtraExtraExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[7])
        # self.gap_error_rate['ExtraExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[8])
        # self.gap_error_rate['ExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[9])
        # self.gap_error_rate['Small'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[10])
        # self.gap_error_rate['Medium'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[11])
        # self.gap_error_rate['Large'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[12])
        # self.gap_error_rate['ExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[13])
        # self.gap_error_rate['ExtraExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[14])

        # # self.gap_diff_from_min['Small'] = fuzz.trimf(self.gap_diff_from_min.universe, [-3, -1.5, 0])
        # # self.gap_diff_from_min['Medium'] = fuzz.trimf(self.gap_diff_from_min.universe, [-1.5, 0, 1.5])
        # # self.gap_diff_from_min['Large'] = fuzz.trimf(self.gap_diff_from_min.universe, [0, 1.5, 2])

        # # setup the 12 output membership functions
        # self.acceleration['ExtraExtraSmall'] = fuzz.trimf(self.acceleration.universe, membership_function_values[15])
        # self.acceleration['ExtraSmall'] = fuzz.trimf(self.acceleration.universe, membership_function_values[16])
        # self.acceleration['Small'] = fuzz.trimf(self.acceleration.universe, membership_function_values[17])
        # self.acceleration['Medium'] = fuzz.trimf(self.acceleration.universe, membership_function_values[18])
        # self.acceleration['Large'] = fuzz.trimf(self.acceleration.universe, membership_function_values[19])
        # self.acceleration['ExtraLarge'] = fuzz.trimf(self.acceleration.universe, membership_function_values[20])
        # self.acceleration['ExtraExtraLarge'] = fuzz.trimf(self.acceleration.universe, membership_function_values[21])

        # # STAGE TWO: DEFINE RULE BASE AND INFERENCE USING SCALED OUTPUT APPROACH
        # rule1 = ctrl.Rule(antecedent=((self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraExtraSmall']) |
        #                               (self.gap_error['ExtraExtraExtraSmall'] & self.gap_error_rate['ExtraExtraExtraSmall'])),
        #                   consequent=self.acceleration['ExtraExtraSmall'])

        # rule2 = ctrl.Rule(antecedent=(self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraExtraSmall']),
        #                   consequent=self.acceleration['ExtraSmall'])

        # rule3 = ctrl.Rule(antecedent=((self.gap_error['Small'] & self.gap_error_rate['ExtraExtraSmall']) |
        #                               (self.gap_error['Medium'] & self.gap_error_rate['ExtraExtraSmall']) |
        #                               (self.gap_error['Large'] & self.gap_error_rate['ExtraExtraSmall']) |
        #                               (self.gap_error['ExtraLarge'] & self.gap_error_rate['ExtraExtraSmall']) |
        #                               (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraSmall']) |
        #                               (self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraSmall']) |
        #                               (self.gap_error['Small'] & self.gap_error_rate['ExtraSmall']) |
        #                               (self.gap_error['Medium'] & self.gap_error_rate['ExtraSmall']) |
        #                               (self.gap_error['Large'] & self.gap_error_rate['ExtraSmall']) |
        #                               (self.gap_error['ExtraLarge'] & self.gap_error_rate['ExtraSmall']) |
        #                               (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['Small']) |
        #                               (self.gap_error['ExtraSmall'] & self.gap_error_rate['Small']) |
        #                               (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['Medium']) |
        #                               (self.gap_error['ExtraSmall'] & self.gap_error_rate['Medium'])),
        #                   consequent=self.acceleration['Small'])

        # rule4 = ctrl.Rule(antecedent=((self.gap_error['Small'] & self.gap_error_rate['Small']) |
        #                               (self.gap_error['Medium'] & self.gap_error_rate['Small']) |
        #                               (self.gap_error['Large'] & self.gap_error_rate['Small']) |
        #                               (self.gap_error['ExtraLarge'] & self.gap_error_rate['Small']) |
        #                               (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['Large']) |
        #                               (self.gap_error['ExtraSmall'] & self.gap_error_rate['Large']) |
        #                               (self.gap_error['Medium'] & self.gap_error_rate['Medium'])),
        #                   consequent=self.acceleration['Medium'])

        # rule5 = ctrl.Rule(antecedent=((self.gap_error['Small'] & self.gap_error_rate['Medium']) |
        #                               (self.gap_error['Large'] & self.gap_error_rate['Medium']) |  # added this here
        #                               (self.gap_error['ExtraLarge'] & self.gap_error_rate['Medium']) |
        #                               (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraLarge']) |
        #                               (self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraLarge']) |
        #                               (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraExtraLarge']) |
        #                               (self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraExtraLarge'])),
        #                   consequent=self.acceleration['Large'])

        # rule6 = ctrl.Rule(antecedent=((self.gap_error['Medium'] & self.gap_error_rate['Large']) |
        #                               (self.gap_error['Large'] & self.gap_error_rate['Large']) |
        #                               (self.gap_error['Small'] & self.gap_error_rate['ExtraLarge']) |
        #                               (self.gap_error['Medium'] & self.gap_error_rate['ExtraLarge'])),
        #                   consequent=self.acceleration['ExtraLarge'])

        # rule7 = ctrl.Rule(antecedent=((self.gap_error['ExtraLarge'] & self.gap_error_rate['Large']) |
        #                               (self.gap_error['Large'] & self.gap_error_rate['ExtraLarge']) |
        #                               (self.gap_error['ExtraLarge'] & self.gap_error_rate['ExtraLarge']) |
        #                               (self.gap_error['Small'] & self.gap_error_rate['ExtraExtraLarge']) |
        #                               (self.gap_error['Medium'] & self.gap_error_rate['ExtraExtraLarge']) |
        #                               (self.gap_error['Large'] & self.gap_error_rate['ExtraExtraLarge']) |
        #                               (self.gap_error['ExtraLarge'] & self.gap_error_rate['ExtraExtraLarge'])),
        #                   consequent=self.acceleration['ExtraExtraLarge'])

        # # rule1.view()

        # SUMO_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])

        # SUMO = ctrl.ControlSystemSimulation(SUMO_control, flush_after_run=100)  # using cache=False did not help

        SUMO.input['gap-error-value'] = vehicle_gap_error
        SUMO.input['gap-error-change-rate-value'] = vehicle_gap_error_rate
        # SUMO.input['gap-diff-from-avg'] = inputs[3]

        SUMO.compute()
        # print(f"skfuzzy number of runs is: {SUMO._run}")

        result = SUMO.output['acceleration-value']

        # SUMO.reset()

        # del inputs, self.acceleration, self.gap_error, self.gap_error_rate, SUMO, SUMO_control, rule1, rule2, rule3, rule4, rule5, rule6, rule7
        # # print("The garbace count in fuzzyHW is: ", gc.get_count())
        # gc.collect()

        return result

    # @profile
    def vehicle_fuzzy(self, vehicle_id, vehicle_gap_error, vehicle_gap_error_rate, SUMO):
        fuzzyOut = float
        acceleration_val = []
        # count = 0
        fuzzyFunction = FuzzyHWClass()
        fuzzyOut = fuzzyFunction.fuzzyHW(vehicle_id, vehicle_gap_error, vehicle_gap_error_rate, SUMO)
        acceleration_val.append(fuzzyOut)
        # print(f"{y_val} and {fuzzyOut}")

        # del fuzzyOut
        # gc.collect()

        return acceleration_val

    def itteration_over_df(df_excel_data, vehicle):
        vehicle_dataframe = df_excel_data[df_excel_data.vehicle_id == vehicle]
        return vehicle_dataframe

    def average(input):
        return sum(input) / len(list)

    # @profile
    def calc_Inputs(self, vehicle_id, previous_vehicles_position, previous_gap, vehicle_position, vehicle_speed, list_vehicle_gap_errors, avgTimeLoss, timeLossChangeRate, SUMO, SUMOLANECHANGE):
        # fuzzyFunction = FuzzyHWClass()
        # constants
        ideal_gap = 1  # second
        vehicle_length = 5  # meters

        # trying to use itteration_over_df() to separate the dataframe per vehicle.
        count = 0

        # create a lits of the inputs to the systems
        # vehicle_gap = previous vehicle position - vehicle length - ego vehicle position
        # unit: seconds
        vehicle_gap = (previous_vehicles_position - vehicle_length - vehicle_position) / vehicle_speed  # updated 1/4/2023
        # if vehicle_velocity > 0:
        #     vehicle_gap_error.append((vehicle_gap/vehicle_velocity)-ideal_gap)
        # else:
        # unit: seconds
        previous_gap_error = previous_gap-ideal_gap
        vehicle_gap_error = vehicle_gap-ideal_gap

        # unit: m/s
        vehicle_gap_error_rate = (vehicle_gap_error-previous_gap_error) # * vehicle_speed  # updated 1/4/2023

        # print(count)
        # print(vehicle_id, vehicle_gap, vehicle_gap_error, vehicle_gap_error_rate)

        if len(list_vehicle_gap_errors) == 0:
            list_vehicle_gap_errors.append(vehicle_gap_error)
            # average_vehicle_gaps = fuzzyFunction.average(list_vehicle_gap_errors)
            min_vehicle_gap = np.nanmin(list_vehicle_gap_errors)
            ego_gap_diff_from_min = vehicle_gap_error - min_vehicle_gap
        else:
            # average_vehicle_gaps = fuzzyFunction.average(list_vehicle_gap_errors)
            min_vehicle_gap = np.nanmin(list_vehicle_gap_errors)
            ego_gap_diff_from_min = vehicle_gap_error - min_vehicle_gap

        # vehicle_acceleration = fuzzyFunction.vehicle_fuzzy(vehicle_id, vehicle_gap_error, vehicle_gap_error_rate, SUMO)
        # fuzzyOut = float
        SUMO.input['gap-error-value'] = vehicle_gap_error
        SUMO.input['gap-error-change-rate-value'] = (vehicle_gap_error_rate * 30)
        # SUMO.input['gap-diff-from-avg'] = inputs[3]

        SUMO.compute()
        # print(f"skfuzzy number of runs is: {SUMO._run}")

        result = SUMO.output['acceleration-value']

        # lane change behavior - FIS
        SUMOLANECHANGE.input['avg-time-loss'] = avgTimeLoss
        SUMOLANECHANGE.input['avg-time-loss-rate'] = timeLossChangeRate

        SUMOLANECHANGE.compute()

        lane_change_decision = round(SUMOLANECHANGE.output['change-lane-decision'])
        # fuzzyOut = fuzzyFunction.fuzzyHW(vehicle_id, vehicle_gap_error, vehicle_gap_error_rate, SUMO)
        vehicle = [vehicle_gap, vehicle_gap_error, vehicle_gap_error_rate, result, lane_change_decision]

        count = count+1
        # del vehicle_acceleration, min_vehicle_gap, list_vehicle_gap_errors, vehicle_gap_error_rate, vehicle_gap_error, \
        #     previous_gap_error, vehicle_gap
        # # print("The garbace count in calc_Inputs is: ", gc.get_count())
        # gc.collect()
        return vehicle

    def run(df_excel_data):
        fuzzyFunction = FuzzyHWClass()
        vehicle_1, vehicle_2, vehicle_3, vehicle_4 = fuzzyFunction.calc_Inputs(df_excel_data)

        return vehicle_1, vehicle_2, vehicle_3, vehicle_4


def save_data(vehicle_id, vehicle_array):
    df_vehicle = pd.DataFrame(vehicle_array)
    filepath = f'./vehicle_{vehicle_id}.csv'
    df_vehicle.to_csv(filepath, index=False)


if __name__ == "__main__":
    # bring in the excel data
    df_excel_data = pd.read_csv('000_fcdout.csv')

    # print(df_excel_data)
    vehicle_1, vehicle_2, vehicle_3, vehicle_4 = FuzzyHWClass.run(df_excel_data)
    vehicle_1_array = np.array(vehicle_1)
    vehicle_2_array = np.array(vehicle_2)
    vehicle_3_array = np.array(vehicle_3)
    vehicle_4_array = np.array(vehicle_4)

    save_data(1, vehicle_1_array)
    save_data(2, vehicle_2_array)
    save_data(3, vehicle_3_array)
    save_data(4, vehicle_4_array)

    plt.plot(range(71), vehicle_1_array[:, 2])
    plt.show()
