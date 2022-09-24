import numpy as np
import matplotlib.pyplot as plt
import random
from fuzzy_asteroids.fuzzy_asteroids import TrainerEnvironment
# from GA.fitnessFunction2 import *
from GA.chromosome import *
from GA.initializationFunctions import *
from GA.selectionFunctions import *
from GA.crossoverFunctions import *
from GA.mutationFunctions import *
from GA.elitismFunctions import *
from GA.continousGeneticAlgorithm import CGA
from HeiTerryController3 import FuzzyController
from sample_score import SampleScore
from shapely import speedups
from GA.chromosome import *
from fuzzy_asteroids.util import Scenario

settings = {
    #"real_time_multiplier": 3,
    "lives": 1,
    "time_limit": 150
}

scenario_ast = Scenario(
    asteroid_states=[{"position": (200, 550), "angle": 84.6, "speed": 25},
                     {"position": (600, 500), "angle": 169.0, "speed": 40},
                     {"position": (700, 200), "angle": 46.8, "speed": 30},
                     {"position": (300, 100), "angle": 135.0, "speed": 50},
                     {"position": (100, 400), "angle": -26.0, "speed": 42},
                     {"position": (500, 200), "angle": -96.0, "speed": 25},
                     ]
)

game = TrainerEnvironment(settings=settings)

myCGA = CGA(NumberOfChrom = 20,
          NumbofGenes = 10,
          maxGen = 10,
          PC = 0.75,
          PM = 0,
          Er = 0.15,
          bounds = game)

myCGA.initialization(asteriodInitialize4)

def AsteriodFitness(chrom, bounds):
    fitness = game.run(controller=FuzzyController(chrom), score=SampleScore(), scenario=scenario_ast)
    return fitness

myCGA.run(selectionFunction = basicSelection,
        crossoverFunction = AsteriodsCrossoverRand1Point2,
        mutationFunction = asteriodMutation,
        fitnessFunction = AsteriodFitness,
        elitismFunction = ElitismTest)

best = myCGA.getBestChromosome()

print("\n\n",best)
