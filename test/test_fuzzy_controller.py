import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd


class FuzzyHWClass:

    def fuzzyHW(self, input):
        # initialize fuzy variables
        self.gap_error = ctrl.Antecedent(np.arange(-1.5, 2.6, 0.1), 'gap-error-value')  # noqa: E501
        self.gap_error_rate = ctrl.Antecedent(np.arange(-6, 6.1, 0.1), 'gap-error-change-rate-value')  # noqa: E501

        # output acceleration
        self.acceleration = ctrl.Consequent(np.arange(-5, 5.1, 0.1), 'acceleration-value')

        # Function for fuzz.trimf(input,left edge, center edge, right edge)
        self.gap_error['ExtraExtraSmall'] = fuzz.trimf(self.gap_error.universe, [-0.6, -0.5, -0.25])  # noqa: E501
        self.gap_error['ExtraSmall'] = fuzz.trimf(self.gap_error.universe, [-0.5, -0.25, 0])
        self.gap_error['Small'] = fuzz.trimf(self.gap_error.universe, [-0.25, 0, 0.25])
        self.gap_error['Medium'] = fuzz.trimf(self.gap_error.universe, [0, 0.25, 0.5])
        self.gap_error['Large'] = fuzz.trimf(self.gap_error.universe, [0.25, 0.5, 1])
        self.gap_error['ExtraLarge'] = fuzz.trimf(self.gap_error.universe, [0.5, 1, 3])
        print(self.gap_error.view())

        self.gap_error_rate['ExtraExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, [-6, -5.36, -2.235])
        self.gap_error_rate['ExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, [-5.36, -2.235, -0.447])
        self.gap_error_rate['Small'] = fuzz.trimf(self.gap_error_rate.universe, [-2.235, -0.447, 0])
        self.gap_error_rate['Medium'] = fuzz.trimf(self.gap_error_rate.universe, [-0.447, 0, 0.447])
        self.gap_error_rate['Large'] = fuzz.trimf(self.gap_error_rate.universe, [0, 0.447, 2.235])
        self.gap_error_rate['ExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, [0.447, 2.235, 5.36])
        self.gap_error_rate['ExtraExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, [2.235, 5.36, 6])
        print(self.gap_error_rate.view())

        # setup the 12 output membership functions
        self.acceleration['ExtraExtraSmall'] = fuzz.trimf(self.acceleration.universe, [-5, -4.572, -3])
        self.acceleration['ExtraSmall'] = fuzz.trimf(self.acceleration.universe, [-4.572, -3, -1.5])
        self.acceleration['Small'] = fuzz.trimf(self.acceleration.universe, [-2.235, -1.5, 0])
        self.acceleration['Medium'] = fuzz.trimf(self.acceleration.universe, [-1.5, 0, 1.5])
        self.acceleration['Large'] = fuzz.trimf(self.acceleration.universe, [0, 1.5, 3])
        self.acceleration['ExtraLarge'] = fuzz.trimf(self.acceleration.universe, [1.5, 3, 4.572])
        self.acceleration['ExtraExtraLarge'] = fuzz.trimf(self.acceleration.universe, [3, 4.572, 5])
        print(self.acceleration.view())

        # STAGE TWO: DEFINE RULE BASE AND INFERENCE USING SCALED OUTPUT APPROACH  # noqa: E501
        rule1 = ctrl.Rule(anticedent=(self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraExtraSmall']),
                          consequent=self.acceleration['ExtraExtraSmall'])

        rule2 = ctrl.Rule(anticedent=(self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraExtraSmall']),
                          consequent=self.acceleration['ExtraSmall'])

        rule3 = ctrl.Rule(anticedent=((self.gap_error['Small'] & self.gap_error_rate['ExtraExtraSmall']) |
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

        rule4 = ctrl.Rule(anticedent=((self.gap_error['Small'] & self.gap_error_rate['Small']) |
                                      (self.gap_error['Medium'] & self.gap_error_rate['Small']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['Small']) |
                                      (self.gap_error['ExtraLarge'] & self.gap_error_rate['Small']) |
                                      (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['Large']) |
                                      (self.gap_error['ExtraSmall'] & self.gap_error_rate['Large'])),
                          consequent=self.acceleration['Medium'])

        rule5 = ctrl.Rule(anticedent=((self.gap_error['Small'] & self.gap_error_rate['Medium']) |
                                      (self.gap_error['Medium'] & self.gap_error_rate['Medium']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['Medium']) |
                                      (self.gap_error['ExtraLarge'] & self.gap_error_rate['Medium']) |
                                      (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraLarge']) |
                                      (self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraLarge']) |
                                      (self.gap_error['ExtraExtraSmall'] & self.gap_error_rate['ExtraExtraLarge']) |
                                      (self.gap_error['ExtraSmall'] & self.gap_error_rate['ExtraExtraLarge'])),
                          consequent=self.acceleration['Large'])

        rule6 = ctrl.Rule(anticedent=((self.gap_error['Medium'] & self.gap_error_rate['Large']) |
                                      (self.gap_error['Large'] & self.gap_error_rate['Large']) |
                                      (self.gap_error['Small'] & self.gap_error_rate['ExtraLarge']) |
                                      (self.gap_error['Medium'] & self.gap_error_rate['ExtraLarge'])),
                          consequent=self.acceleration['ExtraLarge'])

        rule7 = ctrl.Rule(anticedent=((self.gap_error['ExtraLarge'] & self.gap_error_rate['Large']) |
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

        SUMO.input['gap-error-value'] = input[0]
        SUMO.input['gap-error-change-rate-value'] = input[1]

        SUMO.compute()

        result = SUMO.output['acceleration-value']

        return result

    def vehicle_fuzzy(vehicle_inputs):
        fuzzyOut = float
        acceleration_val = []
        # count = 0
        for i in vehicle_inputs:
            # print(i)
            fuzzyFunction = FuzzyHWClass()
            fuzzyOut = fuzzyFunction.fuzzyHW(i)
            acceleration_val.append(fuzzyOut)
            # print(f"{y_val} and {fuzzyOut}")

        return acceleration_val

    def itteration_over_df(df_excel_data, vehicle):
        vehicle_dataframe = df_excel_data[df_excel_data.vehicle_id == vehicle]
        return vehicle_dataframe

    def run(df_excel_data):
        fuzzyFunction = FuzzyHWClass()
        # constants
        ideal_gap = 2
        # calculate the input to the system
        vehicle_0_position = []
        vehicle_1_position = []
        vehicle_2_position = []
        vehicle_3_position = []
        vehicle_4_position = []

        vehicle_1_gap = []
        vehicle_2_gap = []
        vehicle_3_gap = []
        vehicle_4_gap = []

        vehicle_1_gap_error = []
        vehicle_2_gap_error = []
        vehicle_3_gap_error = []
        vehicle_4_gap_error = []

        vehicle_1_velocity = []
        vehicle_2_velocity = []
        vehicle_3_velocity = []
        vehicle_4_velocity = []

        vehicle_1_acceleration = []
        vehicle_2_acceleration = []
        vehicle_3_acceleration = []
        vehicle_4_acceleration = []

        vehicle_1 = []
        vehicle_2 = []
        vehicle_3 = []
        vehicle_4 = []

        # trying to use itteration_over_df() to separate the dataframe per vehicle.

        for i in df_excel_data.index:
            if df_excel_data.vehicle_id[i] == 0:
                vehicle_0_position.append(df_excel_data.vehicle_pos[i])
            elif df_excel_data.vehicle_id[i] == 1:
                vehicle_1_position.append(df_excel_data.vehicle_pos[i])
                vehicle_1_velocity.append(df_excel_data.vehicle_speed[i])
            elif df_excel_data.vehicle_id[i] == 2:
                vehicle_2_position.append(df_excel_data.vehicle_pos[i])
                vehicle_2_velocity.append(df_excel_data.vehicle_speed[i])
            elif df_excel_data.vehicle_id[i] == 3:
                vehicle_3_position.append(df_excel_data.vehicle_pos[i])
                vehicle_3_velocity.append(df_excel_data.vehicle_speed[i])
            elif df_excel_data.vehicle_id[i] == 4:
                vehicle_4_position.append(df_excel_data.vehicle_pos[i])
                vehicle_3_velocity.append(df_excel_data.vehicle_speed[i])
            else:
                print("This data is not recognized")

            # create a lits of the inputs to the systems
            for i in vehicle_0_position:
                vehicle_1_gap.append(vehicle_0_position[i] - vehicle_1_position[i])
                vehicle_1_gap_error.append((vehicle_1_gap/vehicle_1_velocity)-ideal_gap)

                vehicle_2_gap.append(vehicle_1_position[i] - vehicle_2_position[i])
                vehicle_2_gap_error.append((vehicle_2_gap/vehicle_2_velocity)-ideal_gap)

                vehicle_3_gap.append(vehicle_2_position[i] - vehicle_3_position[i])
                vehicle_3_gap_error.append((vehicle_3_gap/vehicle_3_velocity)-ideal_gap)

                vehicle_4_gap.append(vehicle_3_position[i] - vehicle_4_position[i])
                vehicle_4_gap_error.append((vehicle_4_gap/vehicle_4_velocity)-ideal_gap)

                if i >= 1:
                    vehicle_1_gap_error_rate = vehicle_1_velocity/(vehicle_1_gap_error[i]-vehicle_1_gap_error[i-1])
                    vehicle_2_gap_error_rate = vehicle_2_velocity/(vehicle_2_gap_error[i]-vehicle_2_gap_error[i-1])
                    vehicle_3_gap_error_rate = vehicle_3_velocity/(vehicle_3_gap_error[i]-vehicle_3_gap_error[i-1])
                    vehicle_4_gap_error_rate = vehicle_4_velocity/(vehicle_4_gap_error[i]-vehicle_4_gap_error[i-1])
                else:
                    vehicle_1_gap_error_rate = 0
                    vehicle_2_gap_error_rate = 0
                    vehicle_3_gap_error_rate = 0
                    vehicle_4_gap_error_rate = 0

                vehicle_1_acceleration.append(fuzzyFunction.vehicle_fuzzy([vehicle_1_gap_error[i], vehicle_1_gap_error_rate[i]]))
                vehicle_1.append([vehicle_1_gap_error, vehicle_1_gap_error_rate, vehicle_1_acceleration])

                vehicle_2_acceleration.append(fuzzyFunction.vehicle_fuzzy([vehicle_2_gap_error[i], vehicle_2_gap_error_rate[i]]))
                vehicle_2.append([vehicle_2_gap_error, vehicle_2_gap_error_rate, vehicle_2_acceleration])

                vehicle_3_acceleration.append(fuzzyFunction.vehicle_fuzzy([vehicle_3_gap_error[i], vehicle_3_gap_error_rate[i]]))
                vehicle_3.append([vehicle_3_gap_error, vehicle_3_gap_error_rate, vehicle_3_acceleration])

                vehicle_4_acceleration.append(fuzzyFunction.vehicle_fuzzy([vehicle_4_gap_error[i], vehicle_4_gap_error_rate[i]]))
                vehicle_4.append([vehicle_4_gap_error, vehicle_4_gap_error_rate, vehicle_4_acceleration])

        return vehicle_1


if __name__ == "__main__":
    # bring in the excel data
    df_excel_data = pd.read_csv('fcdout.csv')

    print(df_excel_data)
    vehicle_result = FuzzyHWClass.run(df_excel_data)
    print(vehicle_result)
