import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

###### CLASS FOR FUZZY SYSTEM ######

class FIS():
    def __init__(self):
        self.inputs = dict()
        self.outputs = dict()
        self.rulebase = []

    def add_input(self, name, ranger, mfs, mfsname=None):
        # mfs = [[3,4,5], [4,2,4,6], [1,3,5]]
        # ranger = np.arange(0,9,1)
        # name = 'quality'

        # create antecedent
        tempin = ctrl.Antecedent(ranger, name)

        # create membership functions
        if type(mfs) == int:
            tempin.automf(mfs)
        else:
            for e in range(len(mfs)):
                if len(mfs[e]) == 3:
                    tempin[str(e)] = fuzz.trimf(tempin.universe, mfs[e])
                elif len(mfs[e]) == 4:
                    tempin[str(e)] = fuzz.trapmf(tempin.universe, mfs[e])

        # add to input dictionary
        self.inputs.update({name: tempin})

    def add_output(self, name, ranger, mfs, mfsname=None):
        # mfs = [[3,4,5], [4,2,4,6], [1,3,5]]? or just some vlaue for whether to do triangle or custom

        # create consequent
        tempout = ctrl.Consequent(ranger, name)

        # create membership functions
        if type(mfs) == int:
            tempout.automf(mfs)
        else:
            for e in range(len(mfs)):
                if len(mfs[e]) == 3:
                    tempout[str(e)] = fuzz.trimf(tempout.universe, mfs[e])
                elif len(mfs[e]) == 4:
                    tempout[str(e)] = fuzz.trapmf(tempout.universe, mfs[e])

        # add to output dictionary
        self.outputs.update({name: tempout})

    def generate_mamdani_rule(self, input, action, output):
        """inputs = [['quality', 'poor'], ['service', 'poor']]
        action = 'OR'
        outputs = [['tip', 0]]"""

        # save rule
        rule = []
        for e, i in enumerate(input):
            rule.append(self.inputs[input[e][0]][input[e][1]])
        rule_output = [self.outputs[output[0][0]][output[0][1]]]

        inputString = ''
        for e in range(len(rule) - 1):
            # define operator
            if action == 'OR':
                inputString += 'rule[' + str(e) + '] | '
            elif action == 'AND':
                inputString += 'rule[' + str(e) + '] & '
            else:
                raise Exception("enter 'OR' or 'AND' as action")

        # create and save rule
        inputString += 'rule[' + str(len(rule)-1) + ']'
        outputString = 'rule_output[0]'

        self.rulebase.append(eval('ctrl.Rule(' + inputString + ',' + outputString + ')'))

    def create_control_system(self):
        # create control system and simulation
        self.control_system = ctrl.ControlSystem(self.rulebase)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def computing(self, crisp_values):
        """crisp_values = [['quality', 3.2], ['service', 5.3]] """

        # compute final output
        for e in range(len(crisp_values)):
            self.simulation.input[crisp_values[e][0]] = crisp_values[e][1]
        self.simulation.compute()
        for key in self.outputs.keys():
            crisp_output = eval("self.simulation.output['" + key + "']")
        return crisp_output

def main():
    """fizzy = FIS()
    fizzy.add_input('quality', np.arange(0, 10, 2), 3)
    fizzy.add_output('tip', np.arange(0, 10, 2), 3)
    inputs = [['quality', 'poor']]
    action = 'OR'
    output = [['tip', 'poor']]
    fizzy.generate_mamdani_rule_base(inputs, action, output)
    fizzy.create_control_system()
    fizzy.computing([['quality', 3.2]])
    #print(fizzy.inputs['quality'].view())"""
    pass


if __name__ == '__main__':
    main()
