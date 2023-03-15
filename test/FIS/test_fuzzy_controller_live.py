from matplotlib import pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd


class FuzzyHWClass:

    def fuzzyHW(self, vehicle_id, vehicle_gap_error, vehicle_gap_error_rate, membership_function_values, *ego_gap_diff_from_min):
        inputs = [vehicle_id, vehicle_gap_error, vehicle_gap_error_rate]  # , ego_gap_diff_from_min]

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

        # initialize fuzy variables
        self.gap_error = ctrl.Antecedent(np.arange(-2, 3, 0.01), 'gap-error-value')
        self.gap_error_rate = ctrl.Antecedent(np.arange(-6, 10, 0.001), 'gap-error-change-rate-value')
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
        # print(self.gap_error.view())

        self.gap_error_rate['ExtraExtraExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[7])
        self.gap_error_rate['ExtraExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[8])
        self.gap_error_rate['ExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[9])
        self.gap_error_rate['Small'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[10])
        self.gap_error_rate['Medium'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[11])
        self.gap_error_rate['Large'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[12])
        self.gap_error_rate['ExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[13])
        self.gap_error_rate['ExtraExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, membership_function_values[14])

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

        SUMO = ctrl.ControlSystemSimulation(SUMO_control)

        SUMO.input['gap-error-value'] = inputs[1]
        SUMO.input['gap-error-change-rate-value'] = inputs[2]
        # SUMO.input['gap-diff-from-avg'] = inputs[3]

        SUMO.compute()

        result = SUMO.output['acceleration-value']

        return result

    def vehicle_fuzzy(self, vehicle_id, vehicle_gap_error, vehicle_gap_error_rate, ego_gap_diff_from_min, membership_function_values):
        fuzzyOut = float
        acceleration_val = []
        # count = 0
        fuzzyFunction = FuzzyHWClass()
        fuzzyOut = fuzzyFunction.fuzzyHW(vehicle_id, vehicle_gap_error, vehicle_gap_error_rate, membership_function_values, ego_gap_diff_from_min)
        acceleration_val.append(fuzzyOut)
        # print(f"{y_val} and {fuzzyOut}")

        return acceleration_val

    def itteration_over_df(df_excel_data, vehicle):
        vehicle_dataframe = df_excel_data[df_excel_data.vehicle_id == vehicle]
        return vehicle_dataframe

    def average(input):
        return sum(input) / len(list)

    def calc_Inputs(self, vehicle_id, previous_vehicles_position, previous_gap, vehicle_position, vehicle_speed, list_vehicle_gap_errors, membership_function_values):
        fuzzyFunction = FuzzyHWClass()
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
        vehicle_gap_error_rate = (vehicle_gap_error-previous_gap_error) * vehicle_speed  # updated 1/4/2023

        # print(count)
        # print(vehicle_id, vehicle_gap, vehicle_gap_error, vehicle_gap_error_rate)

        if len(list_vehicle_gap_errors) == 0:
            list_vehicle_gap_errors.append(vehicle_gap_error)
            # average_vehicle_gaps = fuzzyFunction.average(list_vehicle_gap_errors)
            min_vehicle_gap = min(list_vehicle_gap_errors)
            ego_gap_diff_from_min = vehicle_gap_error - min_vehicle_gap
        else:
            # average_vehicle_gaps = fuzzyFunction.average(list_vehicle_gap_errors)
            min_vehicle_gap = min(list_vehicle_gap_errors)
            ego_gap_diff_from_min = vehicle_gap_error - min_vehicle_gap

        vehicle_acceleration = fuzzyFunction.vehicle_fuzzy(vehicle_id, vehicle_gap_error, vehicle_gap_error_rate, ego_gap_diff_from_min, membership_function_values)
        vehicle = [vehicle_gap, vehicle_gap_error, vehicle_gap_error_rate, vehicle_acceleration[0]]
        count = count+1

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
