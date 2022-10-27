import numpy as np
import random
from GA.chromosome import *


def asteriodInitialize4(numbChroms, numbGenes, bounds):
    population = []
    for i in range(numbChroms):
        population.append(Chromosome(np.random.randint(3, size=(2, 15)).tolist()))
    return population
# Generic CGA Initialization Function
up = 7
lw = -7
def CGAInitialize(numbChroms,numbGenes,bounds):
    population = []
    for i in range(numbChroms):
        gene = np.zeros(numbGenes)
        for j in range(numbGenes):
            gene[j] = (up - lw)*random.random() + lw
        population.append(Chromosome(gene))

    return population

def createUniformInput(numbInputs, numbMems, lbs, ubs, show):
    inputMatrix = np.array([[]] * numbInputs)
    inputMatrix = inputMatrix.tolist()
    for n in range(numbInputs):
        # translate lower bound to zero
        translb = lbs[n] - lbs[n]

        # translate upper bound respectively
        transup = ubs[n] - lbs[n]

        # divide up range evenly
        div = transup / (numbMems - 1)

        # create membership functions
        for i, mem in enumerate(range(numbMems)):
            a = (div * i) - div + lbs[n] + random.uniform(-1, 1)
            b = div * i + lbs[n] + random.uniform(-1, 1)
            c = (div * i) + div + lbs[n] + random.uniform(-1, 1)
            inputMatrix[n].append([a, b, c])
    if show:
        plotMembershipFunctions(inputMatrix[0])
        plt.show()
        plotMembershipFunctions(inputMatrix[1])
        plt.show()
        plotMembershipFunctions(inputMatrix[2])
    return inputMatrix

def tipInitialize(numbChroms, numbGenes, bounds):
    population = []
    for i in range(numbChroms):
        gene = []
        rb1 = np.random.randint(3, size=28).tolist()
        rb2 = np.random.randint(3, size=28).tolist()
        rb3 = np.random.randint(3, size=8).tolist()
        gene.append(rb1)
        gene.append(rb2)
        gene.append(rb3)
        population.append(Chromosome(gene))
    return population

if __name__ == "__main__":
    asteriodInitialize(10, 7, [1, 2])
