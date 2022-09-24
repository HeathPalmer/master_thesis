import numpy as np
import sys
from fuzzy_tools import CustomFIS
from fuzzy_asteroids.fuzzy_asteroids import AsteroidGame, FuzzyAsteroidGame
from HeiTerryController import FuzzyController
from fuzzy_asteroids.fuzzy_asteroids import TrainerEnvironment
from sample_score import SampleScore

def AsteriodFitness(chrom, game):
    # settings = {
    #     # "graphics_on": True,
    #     # "sound_on": False,
    #     # "frequency": 60,
    #     "real_time_multiplier": 2,
    #     # "lives": 3,
    #     # "prints": True,
    #     # "allow_key_presses": False
    # }
    #
    # #game = TrainerEnvironment(settings=settings)
    print(SampleScore)
    print(FuzzyController(chrom))
    print(game)
    score = game.run(controller=FuzzyController(chrom), score=SampleScore())

    return score.fitness
