import numpy as np
#from chromosome import *
import csv
#from FISClasses import FIS
#from sklearn.metrics import mean_squared_error
import threading
import sys
sys.path.append("..")

from fuzzy_tools import CustomFIS

def AsteriodFitness():
    pass

def fitnessOf1(x):
    print(x+7)

def fuzzyHomeworkFitness(chrom):
    X = chrom.getString()
    X = complex(X[0],X[1])
    function = X**5 + 7*X**4 + 6*X**3 - 4*X**2 - 3*X + 2
    fitness_value = 1/np.abs(function)
    return fitness_value

def readCSV(name):
    with open(name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        arr = []
        for i,row in enumerate(reader):
            if i > 0:
                row = row[0].split(",")
                row = [float(r) for r in row]
                arr.append(row)
        return arr


def tipFitnessFuntion(chrom, bonuds):
    """The rule base should be input as an array assuming you loop through each 1st input (ex. meal_cost)
    and inside that you loop through the other ins and etc... 0 means no correlation, 1 is smallest rule, 2
    is next rule up to the largest rule likely 3 or 5. """
    data = readCSV('tipper_train.csv')
    chromosome = chrom.getString()
    tip_output = []
    for col in data:
        ## FIS food
        F1 = FIS()
        rule_base = chromosome[0]
        ins = [['temp', col[0]], ['flavor', col[1]], ['portion', col[2]]]

        F1.add_input('temp', np.arange(0, 9, 1), 3)
        F1.add_input('flavor', np.arange(0, 9, 1), 3)
        F1.add_input('portion', np.arange(0, 9, 1), 3)
        f1_output = chromosome[3]
        F1.add_output('food', np.arange(0, 9, 1), f1_output)

        v_str = ['temp', 'flavor', 'portion', 'food']
        mfs3 = ['poor', 'average', 'good']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(3):
            for gee in range(3):
                for zoop in range(3):
                    rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]], [v_str[2], mfs3[zoop]]],
                                      ['AND'], [[v_str[3], str(rule_base[(wow * 3 * 3) + (gee * 3) + (zoop + 1)])]]])

        for e in range(len(rules_all)):
            F1.generate_mamdani_rule(rules_all[e][0], rules_all[e][1][0], rules_all[e][2])
        F1.create_control_system()
        food = F1.computing(ins)

        ## FIS for the service
        F2 = FIS()
        rule_base = chromosome[1]
        ins = [['attentive', col[3]], ['friend', col[4]], ['speed', col[5]]]

        mf1, mf2, mf3, mf4 = 3, 3, 3, 3
        F2.add_input('attentive', np.arange(0, 9, 1), mf1)
        F2.add_input('friend', np.arange(0, 9, 1), mf2)
        F2.add_input('speed', np.arange(0, 9, 1), mf3)
        f2_output = chromosome[4]
        F2.add_output('service', np.arange(0, 9, 1), f2_output)

        v_str = ['attentive', 'friend', 'speed', 'service']
        mfs3 = ['poor', 'average', 'good']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(3):
            for gee in range(3):
                for zoop in range(3):
                    rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]], [v_str[2], mfs3[zoop]]],
                                      ['AND'], [[v_str[3], str(rule_base[(wow * 3 * 3) + (gee * 3) + (zoop + 1)])]]])

        for e in range(len(rules_all)):
            F2.generate_mamdani_rule(rules_all[e][0], rules_all[e][1][0], rules_all[e][2])
        F2.create_control_system()
        service = F2.computing(ins)

        ## FIS for final output of tip
        F3 = FIS()
        ins = [['food', food], ['service', service]]
        rule_base = chromosome[2]

        mf1, mf2 = 3, 3
        F3.add_input('food', np.arange(0, 9, 1), mf1)
        F3.add_input('service', np.arange(0, 9, 1), mf2)
        tip_outputMem = chromosome[5]
        F3.add_output('tip', np.arange(0, 25, 1), tip_outputMem)

        v_str = ['food', 'service', 'tip']
        mfs3 = ['poor', 'average', 'good']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(2):
            for gee in range(2):
                rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]]], ['AND'],
                                  [[v_str[2], str(rule_base[(wow * 3) + (gee + 1)])]]])

        for e in range(len(rules_all)):
            F3.generate_mamdani_rule(rules_all[e][0], rules_all[e][1][0], rules_all[e][2])

        F3.create_control_system()
        tip = F3.computing(ins)
        tip_output.append(tip)

    def sse(A, B):
        dif = A.ravel() - B.ravel()
        return np.dot(dif, dif)

    tip_actual = np.asarray([d[6] for d in data])
    fitness = 1 / sse(tip_actual, np.asarray(tip_output))
    # fitness = mean_squared_error(tip_actual, np.asarray(tip_output))

    return fitness


def tipFitnessFuntionSection(chrom, bonuds):
    """The rule base should be input as an array assuming you loop through each 1st input (ex. meal_cost)
    and inside that you loop through the other ins and etc... 0 means no correlation, 1 is smallest rule, 2
    is next rule up to the largest rule likely 3 or 5. """
    data = readCSV('tipper_train.csv')
    data = data[0:10]
    chromosome = chrom.getString()
    tip_output = []
    for col in data:
        ## FIS food
        F1 = FIS()
        rule_base = chromosome[0]
        ins = [['temp', col[0]], ['flavor', col[1]], ['portion', col[2]]]

        F1.add_input('temp', np.arange(0, 9, 1), 3)
        F1.add_input('flavor', np.arange(0, 9, 1), 3)
        F1.add_input('portion', np.arange(0, 9, 1), 3)
        f1_output = chromosome[3]
        F1.add_output('food', np.arange(0, 9, 1), f1_output)

        v_str = ['temp', 'flavor', 'portion', 'food']
        mfs3 = ['poor', 'average', 'good']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(3):
            for gee in range(3):
                for zoop in range(3):
                    rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]], [v_str[2], mfs3[zoop]]],
                                      ['AND'], [[v_str[3], str(rule_base[(wow * 3 * 3) + (gee * 3) + (zoop + 1)])]]])

        for e in range(len(rules_all)):
            F1.generate_mamdani_rule(rules_all[e][0], rules_all[e][1][0], rules_all[e][2])
        F1.create_control_system()
        food = F1.computing(ins)

        ## FIS for the service
        F2 = FIS()
        rule_base = chromosome[1]
        ins = [['attentive', col[3]], ['friend', col[4]], ['speed', col[5]]]

        mf1, mf2, mf3, mf4 = 3, 3, 3, 3
        F2.add_input('attentive', np.arange(0, 9, 1), mf1)
        F2.add_input('friend', np.arange(0, 9, 1), mf2)
        F2.add_input('speed', np.arange(0, 9, 1), mf3)
        f2_output = chromosome[4]
        F2.add_output('service', np.arange(0, 9, 1), f2_output)

        v_str = ['attentive', 'friend', 'speed', 'service']
        mfs3 = ['poor', 'average', 'good']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(3):
            for gee in range(3):
                for zoop in range(3):
                    rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]], [v_str[2], mfs3[zoop]]],
                                      ['AND'], [[v_str[3], str(rule_base[(wow * 3 * 3) + (gee * 3) + (zoop + 1)])]]])

        for e in range(len(rules_all)):
            F2.generate_mamdani_rule(rules_all[e][0], rules_all[e][1][0], rules_all[e][2])
        F2.create_control_system()
        service = F2.computing(ins)

        ## FIS for final output of tip
        F3 = FIS()
        ins = [['food', food], ['service', service]]
        rule_base = chromosome[2]

        mf1, mf2 = 3, 3
        F3.add_input('food', np.arange(0, 9, 1), mf1)
        F3.add_input('service', np.arange(0, 9, 1), mf2)
        tip_outputMem = chromosome[5]
        F3.add_output('tip', np.arange(0, 25, 1), tip_outputMem)

        v_str = ['food', 'service', 'tip']
        mfs3 = ['poor', 'average', 'good']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(2):
            for gee in range(2):
                rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]]], ['AND'],
                                  [[v_str[2], str(rule_base[(wow * 3) + (gee + 1)])]]])

        for e in range(len(rules_all)):
            F3.generate_mamdani_rule(rules_all[e][0], rules_all[e][1][0], rules_all[e][2])

        F3.create_control_system()
        tip = F3.computing(ins)
        tip_output.append(tip)

    def sse(A, B):
        dif = A.ravel() - B.ravel()
        return np.dot(dif, dif)

    tip_actual = np.asarray([d[6] for d in data])
    fitness = 1 / sse(tip_actual, np.asarray(tip_output))
    # fitness = mean_squared_error(tip_actual, np.asarray(tip_output))

    return fitness


fitnessArr = []

def tipFitnessFuntionWorker(chrom, col, tip_actual):
    """The rule base should be input as an array assuming you loop through each 1st input (ex. meal_cost)
    and inside that you loop through the other ins and etc... 0 means no correlation, 1 is smallest rule, 2
    is next rule up to the largest rule likely 3 or 5. """
    chromosome = chrom.getString()
    ## FIS food
    F1 = FIS()
    rule_base = chromosome[0]
    ins = [['temp', col[0]], ['flavor', col[1]], ['portion', col[2]]]

    F1.add_input('temp', np.arange(0, 9, 1), 3)
    F1.add_input('flavor', np.arange(0, 9, 1), 3)
    F1.add_input('portion', np.arange(0, 9, 1), 3)
    f1_output = chromosome[3]
    F1.add_output('food', np.arange(0, 9, 1), f1_output)

    v_str = ['temp', 'flavor', 'portion', 'food']
    mfs3 = ['poor', 'average', 'good']
    # Find a way to automate finding num rules per input earlier and for num inputs
    rules_all = []
    for wow in range(3):
        for gee in range(3):
            for zoop in range(3):
                rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]], [v_str[2], mfs3[zoop]]],
                                  ['AND'], [[v_str[3], str(rule_base[(wow * 3 * 3) + (gee * 3) + (zoop + 1)])]]])

    for e in range(len(rules_all)):
        F1.generate_mamdani_rule(rules_all[e][0], rules_all[e][1][0], rules_all[e][2])
    F1.create_control_system()
    food = F1.computing(ins)

    ## FIS for the service
    F2 = FIS()
    rule_base = chromosome[1]
    ins = [['attentive', col[3]], ['friend', col[4]], ['speed', col[5]]]

    mf1, mf2, mf3, mf4 = 3, 3, 3, 3
    F2.add_input('attentive', np.arange(0, 9, 1), mf1)
    F2.add_input('friend', np.arange(0, 9, 1), mf2)
    F2.add_input('speed', np.arange(0, 9, 1), mf3)
    f2_output = chromosome[4]
    F2.add_output('service', np.arange(0, 9, 1), f2_output)

    v_str = ['attentive', 'friend', 'speed', 'service']
    mfs3 = ['poor', 'average', 'good']
    # Find a way to automate finding num rules per input earlier and for num inputs
    rules_all = []
    for wow in range(3):
        for gee in range(3):
            for zoop in range(3):
                rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]], [v_str[2], mfs3[zoop]]],
                                  ['AND'], [[v_str[3], str(rule_base[(wow * 3 * 3) + (gee * 3) + (zoop + 1)])]]])

    for e in range(len(rules_all)):
        F2.generate_mamdani_rule(rules_all[e][0], rules_all[e][1][0], rules_all[e][2])
    F2.create_control_system()
    service = F2.computing(ins)

    ## FIS for final output of tip
    F3 = FIS()
    ins = [['food', food], ['service', service]]
    rule_base = chromosome[2]

    mf1, mf2 = 3, 3
    F3.add_input('food', np.arange(0, 9, 1), mf1)
    F3.add_input('service', np.arange(0, 9, 1), mf2)
    tip_outputMem = chromosome[5]
    F3.add_output('tip', np.arange(0, 25, 1), tip_outputMem)

    v_str = ['food', 'service', 'tip']
    mfs3 = ['poor', 'average', 'good']
    # Find a way to automate finding num rules per input earlier and for num inputs
    rules_all = []
    for wow in range(2):
        for gee in range(2):
            rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]]], ['AND'],
                              [[v_str[2], str(rule_base[(wow * 3) + (gee + 1)])]]])

    for e in range(len(rules_all)):
        F3.generate_mamdani_rule(rules_all[e][0], rules_all[e][1][0], rules_all[e][2])

    F3.create_control_system()
    tip = F3.computing(ins)
    fitness = (tip_actual - tip) ** 2

    fitnessArr.append(fitness)
    return fitness


class myThread(threading.Thread):
    def __init__(self, chromosome, col, actual):
        self.chromosome = chromosome
        self.col = col
        self.actual = actual

    def run(self):
        fitness = tipFitnessFuntionWorker(self.chromosome, self.col, self.actual)
        return fitness


def threadedFitness(chrom):
    fitnessArr = []
    data = readCSV('tipper_train.csv')
    data = data[0:10]
    threads = []
    for d in data:
        t = threading.Thread(target=tipFitnessFuntionWorker, args=(chrom, d, d[6]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # print(fitnessArr)


def threadFunction(chrom, bounds):
    threadedFitness(chrom)
    return 1/sum(fitnessArr)


def CustomTipFitnessFunction(chrom, bonuds):
    """The rule base should be input as an array assuming you loop through each 1st input (ex. meal_cost)
    and inside that you loop through the other ins and etc... 0 means no correlation, 1 is smallest rule, 2
    is next rule up to the largest rule likely 3 or 5. """
    data = readCSV('tipper_train.csv')
    data = data[0:10]
    chromosome = chrom.getString()
    tip_output = []
    for col in data:
        ## FIS food
        F1 = HeiTerry_FIS()
        rule_base = chromosome[0]
        ins = [['temp', col[0]], ['flavor', col[1]], ['portion', col[2]]]

        F1.add_input('temp', np.arange(0, 1, 0.1), 3)
        F1.add_input('flavor', np.arange(0, 1, 0.1), 3)
        F1.add_input('portion', np.arange(0, 1, 0.1), 3)
        f1_output = [[-3,0,4.5],[3,6,9],[4.5,9,12]]
        F1.add_output('food', np.arange(0, 9, 1), f1_output)

        v_str = ['temp', 'flavor', 'portion', 'food']
        mfs3 = ['poor', 'average', 'good']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(3):
            for gee in range(3):
                for zoop in range(3):
                    rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]], [v_str[2], mfs3[zoop]]],
                                      ['AND'], [v_str[3], str(rule_base[(wow * 3 * 3) + (gee * 3) + (zoop + 1)])]])

        F1.generate_mamdani_rule(rules_all)
        food = F1.compute(ins,'food')

        ## FIS for the service
        F2 = HeiTerry_FIS()
        rule_base = chromosome[1]
        ins = [['attentive', col[3]], ['friend', col[4]], ['speed', col[5]]]

        mf1, mf2, mf3, mf4 = 3, 3, 3, 3
        F2.add_input('attentive', np.arange(0, 1, 0.1), mf1)
        F2.add_input('friend', np.arange(0, 1, 0.1), mf2)
        F2.add_input('speed', np.arange(0, 1, 0.1), mf3)
        f2_output = [[-3,0,4.5],[3,6,9],[4.5,9,12]]
        F2.add_output('service', np.arange(0, 9, 1), f2_output)

        v_str = ['attentive', 'friend', 'speed', 'service']
        mfs3 = ['poor', 'average', 'good']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(3):
            for gee in range(3):
                for zoop in range(3):
                    rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]], [v_str[2], mfs3[zoop]]],
                                      ['AND'], [v_str[3], str(rule_base[(wow * 3 * 3) + (gee * 3) + (zoop + 1)])]])

        F2.generate_mamdani_rule(rules_all)
        service = F2.compute(ins,'service')

        ## FIS for final output of tip
        F3 = HeiTerry_FIS()
        ins = [['food', food], ['service', service]]
        rule_base = chromosome[2]

        mf1, mf2 = 3, 3
        F3.add_input('food', np.arange(0, 9, 1), mf1)
        F3.add_input('service', np.arange(0, 9, 1), mf2)
        tip_outputMem = [[0,10,20],[10,20,30],[20,30,40]]
        F3.add_output('tip', np.arange(0, 40, 1), tip_outputMem)

        v_str = ['food', 'service', 'tip']
        mfs3 = ['poor', 'average', 'good']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(2):
            for gee in range(2):
                rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]]], ['AND'],
                                  [v_str[2], str(rule_base[(wow * 3) + (gee + 1)])]])

        F3.generate_mamdani_rule(rules_all)
        tip = F3.compute(ins,'tip')
        tip_output.append(tip)
    """print(tip_output)
    print('next')"""

    def sse(A, B):
        dif = A.ravel() - B.ravel()
        return np.dot(dif, dif)

    tip_actual = np.asarray([d[6] for d in data])
    print(tip_actual,'ac\n')
    print(np.asarray(tip_output))
    fitness = 1 / sse(tip_actual, np.asarray(tip_output))
    # fitness = mean_squared_error(tip_actual, np.asarray(tip_output))

    return fitness



if __name__ == "__main__":
    chrom = Chromosome([-1.0502,0.2965])
    print(fuzzyHomeworkFitness(chrom))