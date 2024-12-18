from EasyGA import GA, crossover
import numpy as np
import random


class My_GA(GA):
    """Example GA with custom initialize_population method."""

    def initialize_population(self):
        """Custom population initialization with chromosomes
        """
        ga_test = My_GA()

        self.population = self.make_population(
            create_pop()
            for _
            in range(self.population_size)
        )
        print(f"The population is: {self.population}")

    """Example GA with custom crossover_individual_impl method."""

    # def crossover_individual_impl(self, parent_1, parent_2, **weight):

    #     # get gene values
    #     value_iter_1 = parent_1.gene_value_iter
    #     value_iter_2 = parent_2.gene_value_iter

    #     # list of average gene values
    #     value_list = [
    #         (value_1+value_2) // 2
    #         for value_1, value_2
    #         in zip(value_iter_1, value_iter_2)
    #     ]

    #     value_sum = sum(value_list)

    #     # ensure the sum adds up to 100
    #     while value_sum < 100:

    #         # add onto a random gene, if possible
    #         index = random.randrange(len(value_list))
    #         if value_list[index] < 25:
    #             value_list[index] += 1
    #             value_sum += 1

    #     return value_list


def create_pop():
    membershipFunction = []
    for x in range(7):
        outterBounds = [random.uniform(-2, 3), random.uniform(-2, 3)]
        outterBounds.sort()
        peak = random.uniform(outterBounds[0], outterBounds[1])
        membershipFunction.append(outterBounds[0])
        membershipFunction.append(peak)
        membershipFunction.append(outterBounds[1])

    for x in range(8):
        outterBounds = [random.uniform(-6, 10), random.uniform(-6, 10)]
        outterBounds.sort()
        peak = random.uniform(outterBounds[0], outterBounds[1])
        membershipFunction.append(outterBounds[0])
        membershipFunction.append(peak)
        membershipFunction.append(outterBounds[1])

    for x in range(7):
        outterBounds = [random.uniform(-5, 5.1), random.uniform(-5, 5.1)]
        outterBounds.sort()
        peak = random.uniform(outterBounds[0], outterBounds[1])
        membershipFunction.append(outterBounds[0])
        membershipFunction.append(peak)
        membershipFunction.append(outterBounds[1])

    return membershipFunction


# Make a custom ga object
ga = My_GA()
ga.generation_goal = 100
ga.chromosome_length = 66
# Run everything.
ga.crossover_individual_impl = crossover.Crossover.Individual.single_point
while ga.active():
    # Evolve only a certain number of generations
    ga.evolve(1)
    # Print the current generation
    ga.print_generation()
    # Print the best chromosome from that generations population
    ga.print_best_chromosome()
    # If you want to show each population
    # ga.print_population()
    # To divide the print to make it easier to look at
    print('-'*75)
    test = []
    for i in range(len(ga.population)):
        # print(f"The chromosome is: {individual}")
        print(f"The chromosome was: {ga.population[i]}")
        test = create_pop()
        # print([[test_item] for test_item in test])
        for j in range(66):
            ga.population[i][j].value = test[j]
        print(f"New chromosome is {ga.population[i]}")
        ga.population[i].fitness = None
        print(f"New chromosome fitness is {ga.population[i].fitness}")
        # ga.population[i] = res
        # print(f"The chromosome is now: {ga.population[i]}")


print(len(ga.population[5]))
# with the gene.value below, I can pull the values from the chromosome and use them in the membership functions
print(ga.population[5][0].value)
i = 0
new_data = []
while i < len(ga.population[5]):
    new_data.append(ga.population[5][i].value)
    i = i+1
print(new_data)

j = 0
chromosome_array_of_arrays = []
while j < len(new_data):
    chromosome_array_of_arrays.append(new_data[j:j+3])
    j = j + 3
chromosome_array_of_arrays = np.array(chromosome_array_of_arrays)
print(chromosome_array_of_arrays)
