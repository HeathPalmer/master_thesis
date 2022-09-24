import numpy as np
from GA.chromosome import *
import random

up = 7
lw = -7
def basicMutation(chromosome,ngene,PM,bounds):
    child = chromosome.getString()
    for i in range(ngene):
        R = random.random()
        if R < PM:
            child[i] = (up - lw)*random.random() + lw
    c = Chromosome(child)
    return c

def tipMutation(chromosome, ngene, PM, bounds):
    child = chromosome.getString()
    leng = len(child)
    for x in range(leng):
        if x <= 2:
            for i in range(x):
                R = random.random()
                if R < PM:
                    child[x][i] = random.randint(0, 2)
        else:
            R = random.random()
            if R < PM:
                child[x][i][0] = child[x][i][0] - random.uniform(0, 1)
                child[x][i][1] = child[x][i][1] + random.uniform(-1, 1)
                child[x][i][2] = child[x][i][2] + random.uniform(0, 1)
    return Chromosome(child)

def asteriodMutation(chromosome, ngene, PM, bounds):
    child = chromosome.getString()
    leng = len(child)
    for x in range(leng):
        for i in range(x):
            R = random.random()
            if R < PM:
                child[x][i] = random.randint(0, 2)
    return Chromosome(child)

def tipMutationTrial(chromosome, ngene, PM, bounds):
    child = chromosome.getString()
    leng = len(child)
    for x in range(leng):
        if x <= 2:
            for i in range(x):
                R = random.random()
                if R < PM:
                    pt1 = random.randint(0, leng - 2)
                    pt2 = random.randint(pt1, leng - 1)
                    dumb = []
                    for donkey in range(pt2-pt1):
                        guess = random.random()
                        if guess < 0.5:
                            child[x][donkey+pt1] = random.randint(0,1)
                        else:
                            child[x][donkey + pt1] = random.randint(1, 2)
        else:
            R = random.random()
            if R < PM:
                child[x][i][0] = child[x][i][0] - random.uniform(0, 1)
                child[x][i][1] = child[x][i][1] + random.uniform(-1, 1)
                child[x][i][2] = child[x][i][2] + random.uniform(0, 1)
    return Chromosome(child)
