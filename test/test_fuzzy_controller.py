import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


class FuzzyHWClass:

    def fuzzyHW(self, input: float):
        # initialize fuzy variables
        self.gap_error = ctrl.Antecedent(np.linspace(16, 118), 'gap-error-value')  # noqa: E501
        self.gap_error_rate = ctrl.Antecedent(np.linspace(-6, 6), 'gap-error-change-rate-value')  # noqa: E501

        # output acceleration
        self.y = ctrl.Consequent(np.linspace(-5, 5), 'acceleration-value')
        # Function for fuzz.trimf(input,left edge, center edge, right edge)
        self.gap_error['ExtraExtraSmall'] = fuzz.trimf(self.gap_error.universe, [16, 16.764, 33.528])  # noqa: E501
        self.gap_error['ExtraSmall'] = fuzz.trimf(self.gap_error.universe, [16.764, 33.528, 50.292])
        self.gap_error['Small'] = fuzz.trimf(self.gap_error.universe, [33.528, 50.292, 67.056])
        self.gap_error['Medium'] = fuzz.trimf(self.gap_error.universe, [50.292, 67.056, 83.82])
        self.gap_error['Large'] = fuzz.trimf(self.gap_error.universe, [67.056, 83.82, 100.584])
        self.gap_error['ExtraLarge'] = fuzz.trimf(self.gap_error.universe, [83.82, 100.584, 117.48])
        self.gap_error['ExtraExtraLarge'] = fuzz.trimf(self.gap_error.universe, [100.584, 117.48, 118])
        print(self.gap_error.view())

        self.gap_error_rate['ExtraExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, [-6, -5.36, -2.235])  # noqa: E501
        self.gap_error_rate['ExtraSmall'] = fuzz.trimf(self.gap_error_rate.universe, [-5.36, -2.235, -0.447])
        self.gap_error_rate['Small'] = fuzz.trimf(self.gap_error_rate.universe, [-2.235, -0.447, 0])
        self.gap_error_rate['Medium'] = fuzz.trimf(self.gap_error_rate.universe, [-0.447, 0, 0.447])
        self.gap_error_rate['Large'] = fuzz.trimf(self.gap_error_rate.universe, [0, 0.447, 2.235])
        self.gap_error_rate['ExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, [0.447, 2.235, 5.36])
        self.gap_error_rate['ExtraExtraLarge'] = fuzz.trimf(self.gap_error_rate.universe, [2.235, 5.36, 6])
        print(self.gap_error_rate.view())

        # setup the 12 output membership functions
        self.y['ExtraExtraSmall'] = fuzz.trimf(self.y.universe, [-5, -4.572, -3])  # noqa: E501
        self.y['ExtraSmall'] = fuzz.trimf(self.y.universe, [-4.572, -3, -1.5])
        self.y['Small'] = fuzz.trimf(self.y.universe, [-2.235, -1.5, 0])
        self.y['Medium'] = fuzz.trimf(self.y.universe, [-1.5, 0, 1.5])
        self.y['Large'] = fuzz.trimf(self.y.universe, [0, 1.5, 3])
        self.y['ExtraLarge'] = fuzz.trimf(self.y.universe, [1.5, 3, 4.572])
        self.y['ExtraExtraLarge'] = fuzz.trimf(self.y.universe, [3, 4.572, 5])
        print(self.y.view())
        # Function for Right fuzz.trimf(input,left edge, right edge)

        # STAGE TWO: DEFINE RULE BASE AND INFERENCE USING SCALED OUTPUT APPROACH  # noqa: E501
        rule1 = ctrl.Rule(self.gap_error['ExtraExtraSmall'], self.y['ExtraExtraSmall'])
        rule2 = ctrl.Rule(self.gap_error['ExtraSmall'], self.y['ExtraSmall'])
        rule3 = ctrl.Rule(self.gap_error['Small'], self.y['Small'])
        rule4 = ctrl.Rule(self.gap_error['Medium'], self.y['Medium'])
        rule5 = ctrl.Rule(self.gap_error['Large'], self.y['Large'])
        rule6 = ctrl.Rule(self.gap_error['ExtraLarge'], self.y['ExtraLarge'])
        rule7 = ctrl.Rule(self.gap_error['ExtraExtraLarge'], self.y['ExtraExtraLarge'])  # noqa: E501

        # rule1.view()

        HW_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])  # noqa: E501

        HW = ctrl.ControlSystemSimulation(HW_control)

        HW.input['x-value'] = input

        HW.compute()

        result = HW.output['y-value']

        return result

    def run():
        # input to the system
        j = np.linspace(0, 1)
        fuzzyOut = float
        y_val = []
        # count = 0
        for i in j:
            # print(i)
            fuzzyFunction = FuzzyHWClass()
            fuzzyOut = fuzzyFunction.fuzzyHW(i)
            y_val.append(fuzzyOut)
            # print(f"{y_val} and {fuzzyOut}")

        return y_val


if __name__ == "__main__":
    y_result = FuzzyHWClass.run()
    # ideal non-linear function y=x^0.45
    x_values = np.linspace(0, 1)
    # x = 0:0.05:1
    nonLinearFunction = []
    for value in x_values:
        nonLinearFunction.append(value ** 0.45)

    # Plot results from Heath's fuzzy logic
    # fig = plt.figure(figsize=(8, 8))
    # Plotting both the curves simultaneously
    plt.plot(x_values, y_result, color='r', label='Fuzzy')
    plt.plot(x_values, nonLinearFunction, color='g', label='NonLinear')

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.title("Heaths Fuzzy System Result & Actual Result")

    # Adding legend, which helps us recognize the curve according to it's color  # noqa: E501
    plt.legend(['Fuzzy Result', 'y = x^0.45'])

    # To load the display window
    plt.show()
