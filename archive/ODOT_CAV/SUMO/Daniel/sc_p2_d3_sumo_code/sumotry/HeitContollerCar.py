from typing import Tuple, Dict, Any
import numpy as np
from fuzzy_asteroids.fuzzy_controller import ControllerBase, SpaceShip
from fuzzy_tools.fuzzy_c_means import c_means
from fuzzy_tools.CustomFIS import HeiTerry_FIS
from fuzzy_tools.circle_functions import findFISInputs, distanceFormula, inRectangle, findClusterInputs
import math
import socket

class FuzzyController(ControllerBase):
    """
    Class to be used by UC Fuzzy Challenge competitors to create a fuzzy logic controller
    for the Asteroid Smasher game.
    Note: Your fuzzy controller class can be called anything, but must inherit from the
    the ``ControllerBase`` class (imported above)
    Users must define the following:
    1. __init__()
    2. actions(self, ship: SpaceShip, input_data: Dict[str, Tuple])
    By defining these interfaces, this class will work correctly
    """
    def __init__(self, chromosome):
        """
        Create your fuzzy logic controllers and other objects here
        """
        """chromosome = [[1, 0, 2, 0, 2, 1, 1, 0, 1, 2, 1, 0, 0, 0, 1, 1, 2, 2, 1, 0, 1, 2, 0, 0, 1, 2, 0,
                       1, 0, 2, 0, 2, 1, 1, 0, 1, 2, 1, 0, 0, 0, 1, 1, 2, 2, 1, 0, 1, 2, 0, 0, 1, 2, 0],
                      [1, 0, 2, 0, 2, 1, 1, 0, 1, 2, 1, 0, 0, 0, 1, 1, 2, 2, 1, 0, 1, 2, 0, 0, 1, 2, 0,
                       1, 0, 2, 0, 2, 1, 1, 0, 1, 2, 1, 0, 0, 0, 1, 1, 2, 2, 1, 0, 1, 2, 0, 0, 1, 2, 0]]"""
        # Get Chromosome String
        chromosome = chromosome.getString()

        # Shooting FIS
        rule_base = chromosome[0][24:28]
        self.S1 = HeiTerry_FIS()
        self.S1.add_input('average_distance', np.arange(0.0, 1.0, 0.1), [[-180.0, -180.0, 0.0], [-180.0, 0.0, 180.0]])
        self.S1.add_input('invaders', np.arange(0.0, 1.0, 0.1), [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        shoot_mems = [[0.0, 0.1, 1.0],[0.2, 0.5, 0.8] ,[0.0, 0.9, 1.0]]
        self.S1.add_output('shooting', np.arange(0.0, 1.0, 0.1), shoot_mems)
        v_str = ['average_distance', 'invaders', 'shooting']
        mfs3 = ['0', '1']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(2):
            for gee in range(2):
                rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]]], ['AND'], [[v_str[2], str(rule_base[(wow) + gee])]]])

        self.S1.generate_mamdani_rule(rules_all)

        # In 150 radius AVOIDANCE FIS
        self.A1 = HeiTerry_FIS()
        rule_base = chromosome[0][0:24]
        r_h_mems = [[-180, -180, -60], [-60, 0, 60], [60, 180, 180]]
        d_mems = [[0, 0, .7], [.3, 1, 1]]
        c_mems = [[-1, -1, .7], [0, 1, 1]]
        self.A1.add_input('relative_heading', np.arange(-180.0, 180.0, 1.0), r_h_mems)
        self.A1.add_input('distance', np.arange(0.0, 1.0, 0.1), d_mems)
        self.A1.add_input('closure_rate', np.arange(-1.0, 1.0, 0.1), c_mems)
        turn_rate_mems = [[-180.0, -180.0, 0.0], [-180.0, 0.0, 180.0], [0.0, 180.0, 180.0]]
        thrust_mems = [[-1.0, -1.0, 0], [-0.5, 0, 0.5], [0.0, 1.0, 1.0]]
        self.A1.add_output('turn_rate', np.arange(-180.0, 180.0, 1.0), turn_rate_mems)
        self.A1.add_output('thrust', np.arange(-4.0, 4.0, 0.1), thrust_mems)
        v_str = ['relative_heading', 'distance', 'closure_rate', 'turn_rate', 'thrust']
        mfs3 = ['0', '1', '2']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(3):
            for gee in range(2):
                for zoop in range(2):
                    rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]], [v_str[2], mfs3[zoop]]],
                                      ['AND'], [[v_str[3], str(rule_base[(wow * 2 * 2) + (gee * 2) + zoop])], # 2
                                                [v_str[4], str(rule_base[12 + (wow * 2 * 2) + (gee * 2) + zoop])]]]) #12


        self.A1.generate_mamdani_rule(rules_all)

        # CLUSTER AVOIDANCE FIS
        self.C1 = HeiTerry_FIS()
        rule_base = chromosome[0][0:24]
        r_h_mems = [[-180, -180, -60], [-60, 0, 60], [60, 180, 180]]
        self.C1.add_input('relative_heading', np.arange(-180.0, 180.0, 1.0), r_h_mems)
        d_mems = [[0,0,.7], [.3,1,1]]
        self.C1.add_input('distance', np.arange(0.0, 1.0, 0.1), d_mems)
        c_mems = [[-1, -1, .7], [0,1,1]]
        self.C1.add_input('closure_rate', np.arange(-1.0, 1.0, 0.1), c_mems)
        turn_rate_mems = [[-180.0, -180.0, 0.0], [-180.0, 0.0, 180.0], [0.0, 180.0, 180.0]]
        thrust_mems = [[-1.0, -1.0, 0], [-0.5, 0, 0.5], [0.0, 1.0, 1.0]]
        self.C1.add_output('turn_rate', np.arange(-180.0, 180.0, 1.0), turn_rate_mems)
        self.C1.add_output('thrust', np.arange(-4.0, 4.0, 0.1), thrust_mems)

        v_str = ['relative_heading', 'distance', 'closure_rate', 'turn_rate', 'thrust']
        mfs3 = ['0', '1', '2']
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(3):
            for gee in range(2):
                for zoop in range(2):
                    rules_all.append([[[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]], [v_str[2], mfs3[zoop]]],
                                      ['AND'], [[v_str[3], str(rule_base[(wow * 2 * 2) + (gee * 2) + zoop])],
                                                [v_str[4], str(rule_base[12 + (wow * 2 * 2) + (gee * 2) + zoop])]]])

        self.C1.generate_mamdani_rule(rules_all)

    def actions(self, ship: SpaceShip, input_data: Dict[str, Tuple]) -> None:
        """
        Compute control actions of the ship. Perform all command actions via the ``ship``
        argument. This class acts as an intermediary between the controller and the environment.
        The environment looks for this function when calculating control actions for the Ship sprite.
        :param ship: Object to use when controlling the SpaceShip
        :param input_data: Input data which describes the current state of the environment
        """


        #### MAIN ####
        # ship positions
        x, y = ship.position
        sx = x  # [m]
        sy = y  # [m]

        # asteriod positions
        asteriods = []
        for x in input_data['asteroids']:
            asteriods.append([x['position'], x['velocity']])

        # create bounding radius
        circles = []
        ychange = 600
        xchange = 800
        radius = 150
        circles.append([(sx, sy), radius, 1])
        circles.append([(sx + xchange, sy), radius, 0])
        circles.append([(sx, sy + ychange), radius, 0])
        circles.append([(sx + xchange, sy + ychange), radius, 0])
        circles.append([(sx - xchange, sy + ychange), radius, 0])
        circles.append([(sx - xchange, sy), radius, 0])
        circles.append([(sx - xchange, sy - ychange), radius, 0])
        circles.append([(sx, sy - ychange), radius, 0])
        circles.append([(sx + xchange, sy - ychange), radius, 0])

        circles = list(map(lambda a: inRectangle(a), circles))

        # obtain Inputs for the 150 radius AVOIDANCE FIS
        avoidanceFisInputs = []
        for c in circles:
            if c[2] == 1:
                for asteriod in asteriods:
                    if distanceFormula(asteriod[0], c[0]) < c[1]:
                        avoidanceFisInputs.append(findFISInputs(c, ship, asteriod))

        # Obtain inputs for the CLUSTER FIS
        num_asteroids = len(input_data['asteroids'])
        X = np.ndarray((num_asteroids, 2))
        for e in range(num_asteroids):
            X[e] = [input_data['asteroids'][e]['position'][0], input_data['asteroids'][e]['position'][1]]
        try:
            centers = c_means(X, nodes=3)
        except:
            centers = None
        clusterFisInputs = []

        if centers is not None:
            for each_center in centers:
                clusterFisInputs.append(findClusterInputs(ship, each_center))

        # Compute FIS outputs for the AVOIDANCE FIS
        turn_rate_each = []
        thrust_each = []
        for each_asteroid in avoidanceFisInputs:
            ins = [['relative_heading', each_asteroid[1]-180], ['distance', each_asteroid[0]/radius], ['closure_rate', each_asteroid[2]]]
            [turn1, thrust1] = self.A1.compute2Plus(ins, ['turn_rate', 'thrust'])
            turn_rate_each.append(turn1)
            thrust_each.append(thrust1)

        # Compute FIS outputs for the CLUSTER FIS
        for each_cluster in clusterFisInputs:
            ins = [['relative_heading', each_cluster[1]-180], ['distance', each_cluster[0]/500], ['closure_rate', each_cluster[2]]]
            [turn2, thrust2] = self.C1.compute2Plus(ins, ['turn_rate', 'thrust'])
            turn_rate_each.append(turn2)
            thrust_each.append(thrust2)

        # Determine the turn rate by averaging all crisp outputs
        if turn_rate_each:
            ship.turn_rate = sum(turn_rate_each)/len(turn_rate_each)
            thrust = sum(thrust_each)/len(thrust_each)
        else:
            thrust = 0

        # Determine thrust input
        if thrust > 0.075:
            ship.thrust = ship.thrust_range[1]
        elif thrust < -0.075:
            ship.thrust = ship.thrust_range[0]
        else:
            ship.thrust = 0

        # Determine Shooting Output
        distance_total = 0
        invaders = 0
        avoidanceFisInputs = []
        for c in circles:
            if c[2] == 1:
                for asteriod in asteriods:
                    if distanceFormula(asteriod[0], c[0]) < c[1]:
                        avoidanceFisInputs.append(findFISInputs(c, ship, asteriod))
                        invaders += 1
                        distance_total += distanceFormula(asteriod[0], c[0]) / 150

        shootingFisInputs = [distance_total / len(asteriods), invaders / len(asteriods)]

        shoot = self.S1.compute([['average_distance', shootingFisInputs[0]], ['invaders', shootingFisInputs[1]]], 'shooting')

        print(shoot)

        if shoot < 0.45:
            ship.shoot()
