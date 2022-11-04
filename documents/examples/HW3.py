import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


class FuzzyHWClass:

    def fuzzyHW(self, input: float):
        self.x = ctrl.Antecedent(np.linspace(0, 1), 'x-value')
        self.y = ctrl.Consequent(np.linspace(0, 1), 'y-value')
        # Function for fuzz.trimf(input,left edge, center edge, right edge)
        self.x['4ExSmall'] = fuzz.trimf(self.x.universe, [-0.05, 0.05, 0.1])
        self.x['3ExSmall'] = fuzz.trimf(self.x.universe, [0, 0.1, 0.2])
        self.x['ExtraExtraSmall'] = fuzz.trimf(self.x.universe, [0.05, 0.15, 0.25])  # noqa: E501
        self.x['ExtraSmall'] = fuzz.trimf(self.x.universe, [0.1, 0.2, 0.3])
        self.x['Small'] = fuzz.trimf(self.x.universe, [0.2, 0.3, 0.4])
        self.x['SmallMedium'] = fuzz.trimf(self.x.universe, [0.3, 0.4, 0.5])
        self.x['Medium'] = fuzz.trimf(self.x.universe, [0.4, 0.5, 0.6])
        self.x['MediumLarge'] = fuzz.trimf(self.x.universe, [0.5, 0.6, 0.7])
        self.x['Large'] = fuzz.trimf(self.x.universe, [0.6, 0.7, 0.8])
        self.x['ExtraLarge'] = fuzz.trimf(self.x.universe, [0.7, 0.8, 0.9])
        self.x['ExtraExtraLarge'] = fuzz.trimf(self.x.universe, [0.8, 0.9, 1])
        # Function for Right fuzz.trimf(input,left edge, right edge)
        self.x['3ExLarge'] = fuzz.trimf(self.x.universe, [0.9, 1, 1.05])

        # setup the 12 output membership functions
        self.y['4ExSmall'] = fuzz.trimf(self.y.universe, [-0.05, 0.15, 0.25])
        self.y['3ExSmall'] = fuzz.trimf(self.y.universe, [0.25, 0.35, 0.45])
        self.y['ExtraExtraSmall'] = fuzz.trimf(self.y.universe, [0.3, 0.4, 0.5])  # noqa: E501
        self.y['ExtraSmall'] = fuzz.trimf(self.y.universe, [0.4, 0.5, 0.6])
        self.y['Small'] = fuzz.trimf(self.y.universe, [0.5, 0.6, 0.7])
        self.y['SmallMedium'] = fuzz.trimf(self.y.universe, [0.6, 0.7, 0.8])
        self.y['Medium'] = fuzz.trimf(self.y.universe, [0.65, 0.75, 0.85])
        self.y['MediumLarge'] = fuzz.trimf(self.y.universe, [0.7, 0.8, 0.9])
        self.y['Large'] = fuzz.trimf(self.y.universe, [0.75, 0.85, 0.95])
        self.y['ExtraLarge'] = fuzz.trimf(self.y.universe, [0.8, 0.9, 1])
        self.y['ExtraExtraLarge'] = fuzz.trimf(self.y.universe, [0.9, 0.95, 1])
        # Function for Right fuzz.trimf(input,left edge, right edge)
        self.y['3ExLarge'] = fuzz.trimf(self.y.universe, [0.95, 0.95, 1.05])

        # STAGE TWO: DEFINE RULE BASE AND INFERENCE USING SCALED OUTPUT APPROACH  # noqa: E501
        rule1 = ctrl.Rule(self.x['4ExSmall'], self.y['4ExSmall'])
        rule2 = ctrl.Rule(self.x['3ExSmall'], self.y['3ExSmall'])
        rule3 = ctrl.Rule(self.x['ExtraExtraSmall'], self.y['ExtraExtraSmall'])
        rule4 = ctrl.Rule(self.x['ExtraSmall'], self.y['ExtraSmall'])
        rule5 = ctrl.Rule(self.x['Small'], self.y['Small'])
        rule6 = ctrl.Rule(self.x['SmallMedium'], self.y['SmallMedium'])
        rule7 = ctrl.Rule(self.x['Medium'], self.y['Medium'])
        rule8 = ctrl.Rule(self.x['MediumLarge'], self.y['MediumLarge'])
        rule9 = ctrl.Rule(self.x['Large'], self.y['Large'])
        rule10 = ctrl.Rule(self.x['ExtraLarge'], self.y['ExtraLarge'])
        rule11 = ctrl.Rule(self.x['ExtraExtraLarge'], self.y['ExtraExtraLarge'])  # noqa: E501
        rule12 = ctrl.Rule(self.x['3ExLarge'], self.y['3ExLarge'])

        # rule1.view()

        HW_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12])  # noqa: E501

        HW = ctrl.ControlSystemSimulation(HW_control)

        HW.input['x-value'] = input

        HW.compute()

        result = HW.output['y-value']

        return result

    def run():
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
