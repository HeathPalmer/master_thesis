import numpy as np
from GA.chromosome import *
import random

up = 7
lw = -7

def basicCrossover(parent1,parent2,nGenes,PC):
    child1 = np.zeros(nGenes)
    child2 = np.zeros(nGenes)
    parent1 = parent1.getString()
    parent2 = parent2.getString()
    for k in range(nGenes):
        beta = random.random()
        child1[k] = beta*parent1[k] + (1-beta)*parent2[k]
        child2[k] = (1-beta)*parent1[k] + beta*parent2[k]

    rand = random.random()
    if rand <= PC:
        parent1 = Chromosome(child1)
    else:
        parent1 = Chromosome(parent1)

    rand = random.random()
    if rand <= PC:
        parent2 = Chromosome(child2)
    else:
        parent2 = Chromosome(parent2)
    
    #print(parent2,"\n",parent1)
    return parent1, parent2

def AsteriodsCrossoverRand1Point(parent1, parent2, nGenes, PC, bounds):
    parent1 = parent1.getString()
    p1_rb1 = parent1[0]
    p1_rb2 = parent1[1]

    parent2 = parent2.getString()
    p2_rb1 = parent2[0]
    p2_rb2 = parent2[1]
    child1 = []
    child2 = []

    def randPointCross(string1, string2):
        leng = len(string1)
        ranInt = random.randint(0, leng - 2)
        ranInt2 = random.randint(ranInt, leng - 1)
        child1 = string1[0:ranInt] + string2[ranInt:ranInt2] + string1[ranInt2:leng]
        child2 = string2[0:ranInt] + string1[ranInt:ranInt2] + string2[ranInt2:leng]
        return child1, child2


    # crossover rule base
    c1_rb1, c2_rb1 = randPointCross(p1_rb1, p2_rb1)
    c1_rb2, c2_rb2 = randPointCross(p1_rb2, p2_rb2)

    child1 = [c1_rb1, c1_rb2]
    child2 = [c2_rb1, c2_rb2]

    rand = random.random()
    if rand <= PC:
        child1 = Chromosome(child1)
    else:
        child1 = Chromosome(parent1)

    rand = random.random()
    if rand <= PC:
        child2 = Chromosome(child2)
    else:
        child2 = Chromosome(parent2)

    return child1, child2

def AsteriodsCrossoverRand1Point2(parent1, parent2, nGenes, PC, bounds):
    parent1 = parent1.getString()
    p1_rb1 = parent1[0]
    #p1_rb2 = parent1[1]

    parent2 = parent2.getString()
    p2_rb1 = parent2[0]
    #p2_rb2 = parent2[1]
    child1 = []
    child2 = []

    def randPointCross(string1, string2):
        leng = len(string1)
        ranInt = random.randint(0, leng - 2)
        ranInt2 = random.randint(ranInt, leng - 1)
        child1 = string1[0:ranInt] + string2[ranInt:ranInt2] + string1[ranInt2:leng]
        child2 = string2[0:ranInt] + string1[ranInt:ranInt2] + string2[ranInt2:leng]
        return child1, child2


    # crossover rule base
    c1_rb1, c2_rb1 = randPointCross(p1_rb1, p2_rb1)
    #c1_rb2, c2_rb2 = randPointCross(p1_rb2, p2_rb2)

    child1 = [c1_rb1]
    child2 = [c2_rb1]

    rand = random.random()
    if rand <= PC:
        child1 = Chromosome(child1)
    else:
        child1 = Chromosome(parent1)

    rand = random.random()
    if rand <= PC:
        child2 = Chromosome(child2)
    else:
        child2 = Chromosome(parent2)

    return child1, child2

def tipCrossoverRand1Point(parent1, parent2, nGenes, PC, bonuds):
    parent1 = parent1.getString()
    p1_rb1 = parent1[0]
    p1_rb2 = parent1[1]
    p1_rb3 = parent1[2]

    parent2 = parent2.getString()
    p2_rb1 = parent2[0]
    p2_rb2 = parent2[1]
    p2_rb3 = parent2[2]
    child1 = []
    child2 = []

    def randPointCross(string1, string2):
        leng = len(string1)
        ranInt = random.randint(0, leng - 2)
        ranInt2 = random.randint(ranInt, leng - 1)
        child1 = string1[0:ranInt] + string2[ranInt:ranInt2] + string1[ranInt2:leng]
        child2 = string2[0:ranInt] + string1[ranInt:ranInt2] + string2[ranInt2:leng]
        return child1, child2


    # crossover rule base
    c1_rb1, c2_rb1 = randPointCross(p1_rb1, p2_rb1)
    c1_rb2, c2_rb2 = randPointCross(p1_rb2, p2_rb2)
    c1_rb3, c2_rb3 = randPointCross(p1_rb3, p2_rb3)

    child1 = [c1_rb1, c1_rb2, c1_rb3]
    child2 = [c2_rb1, c2_rb2, c2_rb3]

    rand = random.random()
    if rand <= PC:
        child1 = Chromosome(child1)
    else:
        child1 = Chromosome(parent1)

    rand = random.random()
    if rand <= PC:
        child2 = Chromosome(child2)
    else:
        child2 = Chromosome(parent2)

    return child1, child2

if __name__ == "__main__":
    chrom = Chromosome([0.205,0.1106])
    chrom2 = Chromosome([0.3,0.6])
    p1,p2 = basicCrossover(chrom,chrom2,2,1)
    print(p1,p2)