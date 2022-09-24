import numpy as np
import matplotlib as mpl
import random
from GA.fitnessFunctions import *
from GA.chromosome import *
from GA.initializationFunctions import *
from GA.selectionFunctions import *
from GA.crossoverFunctions import *
from GA.mutationFunctions import *
from GA.elitismFunctions import *
import threading

class CGA:
    def __init__(self,NumberOfChrom,NumbofGenes,maxGen,PC,PM,Er,bounds):
        self.numbChroms = NumberOfChrom
        self.numbGenes = NumbofGenes
        self.maxGen = maxGen
        self.PC = PC
        self.PM = PM
        self.population = []
        self.Er = Er
        self.bounds = bounds

    def initialization(self, initializationFunction):
        self.population = initializationFunction(self.numbChroms,self.numbGenes, self.bounds)
        for chrom in self.population:
            print(chrom,"\n")

    def fitness(self, fitnessFunction):
        fitnessFunction(3)
        print("goodbye")
        pass

    def selection(self, selectionFunction):
        pass

    def crossover(self):
        rand = random.random()
        if rand > self.PC:
            return True
        else:
            return False
        pass

    def mutation(self,mutationFunction):
        rand = random.random()
        if rand > self.PM:
            return True
        else:
            return False
            
    def run(self,selectionFunction,crossoverFunction,fitnessFunction,mutationFunction,elitismFunction):
        for i in range(self.maxGen):
            print('Generation: ' + str(i))
            best = self.getBestChromosome()
            # best = best.getFitness()
            # print('Best Fitness: ' + str(best))
            print(best)
            # obtain fitness values
            for chrom in self.population:
                chrom.updateFitness(fitnessFunction(chrom, self.bounds))
            newPop = []
            a =np.arange(0,self.numbChroms,2)
            a = a.tolist()
            for k in a:
                # crossover
                parent1, parent2 = selectionFunction(self.population)
                child1, child2 = crossoverFunction(parent1, parent2,self.numbGenes,self.PC,self.bounds)
                
                # mutation
                child1 = mutationFunction(child1, self.numbGenes, self.PM, self.bounds)
                child2 = mutationFunction(child2, self.numbGenes, self.PM, self.bounds)

                newPop.append(child1)
                newPop.append(child2)

            # update fitness values
            for chrom in newPop:
                chrom.updateFitness(fitnessFunction(chrom , self.bounds))
            
            # elitism
            newPop = elitismFunction(self.population, newPop, self.Er)

            self.population = newPop
        # for chrom in self.population:
        #     print(chrom,"\n")
        #print(self.population[0],self.population[1],self.population[2],self.population[3],,self.population[2])                

    def getBestChromosome(self):
        fitnessArr = np.array(list(map(lambda a: a.getFitness(), self.population)))
        norm_idx = np.argsort(fitnessArr)
        norm_idx = np.flip(norm_idx,0)
        bestChrom = self.population[norm_idx[0]]
        return bestChrom

class CGASuperThread:
    def __init__(self, NumberOfChrom, NumbofGenes, maxGen, PC, PM, Er, bounds):
        self.numbChroms = NumberOfChrom
        self.numbGenes = NumbofGenes
        self.maxGen = maxGen
        self.PC = PC
        self.PM = PM
        self.population = []
        self.Er = Er
        self.bounds = bounds

    def initialization(self, initializationFunction):
        self.population = initializationFunction(self.numbChroms, self.numbGenes, self.bounds)
        for chrom in self.population:
            print(chrom, "\n")

    def fitness(self, fitnessFunction):
        fitnessFunction(3)
        print("goodbye")
        pass

    def selection(self, selectionFunction):
        pass

    def crossover(self):
        rand = random.random()
        if rand > self.PC:
            return True
        else:
            return False
        pass

    def mutation(self, mutationFunction):
        rand = random.random()
        if rand > self.PM:
            return True
        else:
            return False

    def run(self, selectionFunction, crossoverFunction, fitnessFunction, mutationFunction, elitismFunction):
        for i in range(self.maxGen):
            print('Generation: ' + str(i))
            best = self.getBestChromosome()
            best = best.getFitness()
            print('Best Fitness: ' + str(best))

            # thread Functions
            def ChromThread(chrom):
                chrom.updateFitness(fitnessFunction(chrom, self.bounds))

            def mainThread():
                # crossover
                parent1, parent2 = selectionFunction(self.population)
                child1, child2 = crossoverFunction(parent1, parent2, self.numbGenes, self.PC, self.bounds)

                # mutation
                child1 = mutationFunction(child1, self.numbGenes, self.PM, self.bounds)
                child2 = mutationFunction(child2, self.numbGenes, self.PM, self.bounds)

                newPop.append(child1)
                newPop.append(child2)

            # obtain fitness values
            threads = []
            for chrom in self.population:
                t = threading.Thread(target=ChromThread, args=(chrom,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            newPop = []
            a = np.arange(0, self.numbChroms, 2)
            a = a.tolist()

            mainThreads = []
            for k in a:
                mt = threading.Thread(target=mainThread)
                mainThreads.append(mt)
                mt.start()

            for mt in mainThreads:
                mt.join()

            # update fitness values
            threads = []
            for chrom in newPop:
                t = threading.Thread(target=ChromThread, args=(chrom,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
                #chrom.updateFitness(fitnessFunction(chrom, self.bounds))

            # elitism
            newPop = elitismFunction(self.population, newPop, self.Er)

            self.population = newPop
        # for chrom in self.population:
        #     print(chrom,"\n")
        # print(self.population[0],self.population[1],self.population[2],self.population[3],,self.population[2])

    def getBestChromosome(self):
        fitnessArr = np.array(list(map(lambda a: a.getFitness(), self.population)))
        norm_idx = np.argsort(fitnessArr)
        norm_idx = np.flip(norm_idx,0)
        bestChrom = self.population[norm_idx[0]]
        return bestChrom


class CGAThread:
    def __init__(self, NumberOfChrom, NumbofGenes, maxGen, PC, PM, Er, bounds):
        self.numbChroms = NumberOfChrom
        self.numbGenes = NumbofGenes
        self.maxGen = maxGen
        self.PC = PC
        self.PM = PM
        self.population = []
        self.Er = Er
        self.bounds = bounds

    def initialization(self, initializationFunction):
        self.population = initializationFunction(self.numbChroms, self.numbGenes, self.bounds)
        for chrom in self.population:
            print(chrom, "\n")

    def fitness(self, fitnessFunction):
        fitnessFunction(3)
        print("goodbye")
        pass

    def selection(self, selectionFunction):
        pass

    def crossover(self):
        rand = random.random()
        if rand > self.PC:
            return True
        else:
            return False
        pass

    def mutation(self, mutationFunction):
        rand = random.random()
        if rand > self.PM:
            return True
        else:
            return False

    def run(self, selectionFunction, crossoverFunction, fitnessFunction, mutationFunction, elitismFunction):
        for i in range(self.maxGen):
            print('Generation: ' + str(i))
            best = self.getBestChromosome()
            #best = best.getFitness()
            #print('Best Fitness: ' + str(best))
            #bestString = best.getString()
            print(best)
            # obtain fitness values
            for chrom in self.population:
                chrom.updateFitness(fitnessFunction(chrom, self.bounds))
            newPop = []
            a = np.arange(0, self.numbChroms, 2)
            a = a.tolist()

            for k in a:
                # crossover
                parent1, parent2 = selectionFunction(self.population)
                child1, child2 = crossoverFunction(parent1, parent2, self.numbGenes, self.PC, self.bounds)

                # mutation
                child1 = mutationFunction(child1, self.numbGenes, self.PM, self.bounds)
                child2 = mutationFunction(child2, self.numbGenes, self.PM, self.bounds)

                newPop.append(child1)
                newPop.append(child2)

            def ChromThread(chrom):
                chrom.updateFitness(fitnessFunction(chrom, self.bounds))

            # update fitness values
            threads = []
            for chrom in newPop:
                t = threading.Thread(target=ChromThread, args=(chrom,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
                #chrom.updateFitness(fitnessFunction(chrom, self.bounds))

            # elitism
            newPop = elitismFunction(self.population, newPop, self.Er)

            self.population = newPop
        # for chrom in self.population:
        #     print(chrom,"\n")
        # print(self.population[0],self.population[1],self.population[2],self.population[3],,self.population[2])

    def getBestChromosome(self):
        fitnessArr = np.array(list(map(lambda a: a.getFitness(), self.population)))
        norm_idx = np.argsort(fitnessArr)
        norm_idx = np.flip(norm_idx,0)
        bestChrom = self.population[norm_idx[0]]
        return bestChrom


# Test
if __name__ == "__main__":
    #fitnessOf1()
    # CGA = CGA(NumberOfChrom = 30,
    #           NumbofGenes = 2,
    #           maxGen = 2000,
    #           PC = 0.85,
    #           PM = 0.15,
    #           Er = 0.2)

    # CGA.initialization(CGAInitialize)
    # CGA.run(selectionFunction = basicSelection,
    #         crossoverFunction = basicCrossover,
    #         mutationFunction = basicMutation,
    #         fitnessFunction = fuzzyHomeworkFitness,
    #         elitismFunction = basicElitism)
    # best = CGA.getBestChromosome()
        pass
