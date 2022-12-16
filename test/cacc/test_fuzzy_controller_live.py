from matplotlib import pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd


class FuzzyHWClass:

    def fuzzyHW(self, vehicle_id, vehicle_gap_error, vehicle_gap_error_rate):
        inputs = [vehicle_id, vehicle_gap_error, vehicle_gap_error_rate]
        # print(inputs)

        # initialize fuzy variables
        self.gap_error = ctrl.Antecedent(np.arange(-2, 3, 0.01), 'gap-error-value')  # noqa: E501
        self.gap_error_rate = ctrl.Antecedent(np.arange(-6, 10, 0.001), 'gap-error-change-rate-value')  # noqa: E501

        # output acceleration
        self.acceleration = ctrl.Consequent(np.arange(-5, 5.1, 0.001), 'acceleration-value')

        # Function for fuzz.trimf(input,left edge, center edge, right edge)
        self.gap_error['ExtraExtraExtraSmall'] = fuzz.trimf(self.gap_error.universe, [-2, -1, -0.5])
        self.gap_error['ExtraExtraSmall'] = fuzz.trimf(self.gap_error.universe, [-0.6, -0.5, -0.25])
        self.gap_error['ExtraSmall'] = fuzz.trimf(self.gap_error.universe, [-0.5, -0.25, 0])
        self.gap_error['Small'] = fuzz.trimf(self.gap_error.universe, [-0.25, 0, 0.25])
        self.gap_error['Medium'] = fuzz.trimf(self.gap_error.universe, [0, 0.5, 1])
        self.gap_error['Large'] = fuzz.trimf(self.gap_error.universe, [0.5, 1, 1.5])
        self.gap_error['ExtraLarge'] = fuzz.trimf(self.gap_error.universe, [1, 1.5, 3])
        # print(self.gap_error.view())

        self.gap_error_rate['ExtraExtraExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, [-10, -7.5, -5.6])
        self.gap_error_rate['ExtraExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, [-6, -5.36, -2.235])
        self.gap_error_rate['ExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, [-5.36, -2.235, -0.447])
        self.gap_error_rate['Small'] = fuzz.trimf(self.gap_error_rate.universe, [-10, -2.235, 0])
        self.gap_error_rate['Medium'] = fuzz.trimf(self.gap_error_rate.universe, [-0.447, 0, 0.447])
        self.gap_error_rate['Large'] = fuzz.trimf(self.gap_error_rate.universe, [0, 0.447, 2.235])
        self.gap_error_rate['ExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, [0.447, 2.235, 5.36])
        self.gap_error_rate['ExtraExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, [2.235, 5.36, 10])

        # setup the 12 output membership functions
        self.acceleration['ExtraExtraSmall'] = fuzz.trimf(self.acceleration.universe, [-5, -4.572, -3])
        self.acceleration['ExtraSmall'] = fuzz.trimf(self.acceleration.universe, [-4.572, -3, -1.5])
        self.acceleration['Small'] = fuzz.trimf(self.acceleration.universe, [-2.235, -1.5, 0])
        self.acceleration['Medium'] = fuzz.trimf(self.acceleration.universe, [-1.5, 0, 1.5])
        self.acceleration['Large'] = fuzz.trimf(self.acceleration.universe, [0, 1.5, 3])
        self.acceleration['ExtraLarge'] = fuzz.trimf(self.acceleration.universe, [1.5, 3, 4.572])
        self.acceleration['ExtraExtraLarge'] = fuzz.trimf(self.acceleration.universe, [3, 4.572, 5])

        # STAGE TWO: DEFINE RULE BASE AND INFERENCE USING SCALED OUTPUT APPROACH  # noqa: E501
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
                                      (self.gap_error['ExtraSmall'] & self.gap_error_rate['Large'] |
                                      (self.gap_error['Medium'] & self.gap_error_rate['Medium']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['Medium']))),
                          consequent=self.acceleration['Medium'])

        rule5 = ctrl.Rule(antecedent=((self.gap_error['Small'] & self.gap_error_rate['Medium']) |
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

        SUMO.compute()

        result = SUMO.output['acceleration-value']
        # print(result)

        return result

    def vehicle_fuzzy(self, vehicle_id, vehicle_gap_error, vehicle_gap_error_rate):
        fuzzyOut = float
        acceleration_val = []
        # count = 0
        fuzzyFunction = FuzzyHWClass()
        fuzzyOut = fuzzyFunction.fuzzyHW(vehicle_id, vehicle_gap_error, vehicle_gap_error_rate)
        acceleration_val.append(fuzzyOut)
        # print(f"{y_val} and {fuzzyOut}")

        return acceleration_val

    def itteration_over_df(df_excel_data, vehicle):
        vehicle_dataframe = df_excel_data[df_excel_data.vehicle_id == vehicle]
        return vehicle_dataframe

    def calc_Inputs(self, vehicle_id, previous_vehicles_position, previous_gap, vehicle_position, vehicle_speed):
        fuzzyFunction = FuzzyHWClass()
        # constants
        ideal_gap = 1
        vehicle_length = 5
        # calculate the input to the system

        vehicle_gap = []

        vehicle_gap_error = []

        vehicle_gap_error_rate = []

        vehicle_velocity = []

        vehicle = []

        # trying to use itteration_over_df() to separate the dataframe per vehicle.
        count = 0

        # create a lits of the inputs to the systems
        # vehicle_gap = previous vehicle position - vehicle length - ego vehicle position
        vehicle_gap.append(previous_vehicles_position - vehicle_length - vehicle_position)

        if vehicle_velocity > 0:
            vehicle_gap_error.append((vehicle_gap/vehicle_velocity)-ideal_gap)
        else:
            if vehicle_gap-previous_gap > 0:
                vehicle_gap_error.append((vehicle_gap/(vehicle_gap-previous_gap))-ideal_gap)
            else:
                vehicle_gap_error.append(0)

        vehicle_gap_error_rate.append((vehicle_gap_error-vehicle_gap_error) * (vehicle_velocity - vehicle_velocity))

        print(count)
        print(vehicle_id, vehicle_gap, vehicle_gap_error, vehicle_gap_error_rate)

        vehicle_acceleration = fuzzyFunction.vehicle_fuzzy(vehicle_id, vehicle_gap_error, vehicle_gap_error_rate)
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
