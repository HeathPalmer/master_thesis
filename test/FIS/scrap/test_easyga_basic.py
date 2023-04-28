import csv
from EasyGA import GA
from math import ceil
import os
import random


# Create the Genetic algorithm.

# class MY_GA(GA):
#     """Example GA with custom parent_selection_impl method."""

#     def parent_selection_impl(self):
#         """Custom parent selection method which selects the best for half of the mating pool
#         and then randomly for the rest of the mating pool."""
#         print("Custom parent selection")

#         mating_pool_size = ceil(len(ga.population)*ga.parent_ratio)
#         ga.population.mating_pool = ga.population[:mating_pool_size//2]  # Select the best half
#         for _ in range((mating_pool_size+1)//2):  # Select the random half
#             ga.population.add_parent(random.choice(ga.population))  # Random chromosome from the population


# Custom implementation of the survivor selection method.
def survivor_selection_impl(ga):
    needed_amount = len(ga.population) - len(ga.population.next_population)
    print(f"population: {ga.population}")
    print(f"needed amount: {needed_amount}")
    print(f"next generation is: {ga.population.next_population}")
    # Save the best chromosome
    if needed_amount > 0:
        best_chromosome = ga.population[0]
        ga.population.add_child(best_chromosome)
        needed_amount -= 1
        print(f"population: {ga.population}")
        print(f"needed amount: {needed_amount}")
        print(f"next generation is: {ga.population.next_population}")
        if needed_amount <= 0:
            return

        # Loop through the population
        for chromosome in ga.population:
            print(f"the chomosome fitness is: {chromosome.fitness}")
            if chromosome.fitness <= 4:
                new_chrom = []
                for i in range(len(chromosome)):
                    new_chrom.append(random.randint(0, 10))
                    # i += 1
                print(f"new chrom is: {new_chrom}")
                ga.population.add_child(new_chrom)
                needed_amount -= 1
                if needed_amount <= 0:
                    break
            # Add chromosome if any of the genes are different
            if any(best_gene != gene for best_gene, gene in zip(best_chromosome, chromosome)):
                ga.population.add_child(chromosome)
                needed_amount -= 1
                print(f"population: {ga.population}")
                print(f"needed amount: {needed_amount}")
                print(f"next generation is: {ga.population.next_population}")

                # Stop if enough chromosomes survive
                if needed_amount <= 0:
                    break


# Create the Genetic algorithm.
ga = GA()

# Setting the survivor selection method.
ga.survivor_selection_impl = survivor_selection_impl
ga.parent_ratio = 0.1

ga.database_name = 'database.db'
if os.path.isfile(ga.database_name):
    os.unlink(ga.database_name)

generation_information_file = 'ga_information.csv'
if os.path.isfile(generation_information_file):
    os.unlink(generation_information_file)

# Run everything.
while ga.active():
    generation_info = []
    # Evolve only a certain number of generations
    ga.evolve(1)
    # Print the current generation
    ga.print_generation()
    # Print the best chromosome from that generations population
    # ga.print_best_chromosome()
    # If you want to show each population
    # ga.print_population()
    # To divide the print to make it easier to look at
    print('-'*75)

    current_gen = ga.current_generation
    current_best_fitness = ga.population[0].fitness
    currenet_best_chromosome = ga.population[0]
    current_population_fitness = []
    for i in range(len(ga.population)):
        current_population_fitness.append(ga.population[i].fitness)
    current_population = ga.population

    generation_info = [current_gen, current_best_fitness, currenet_best_chromosome, current_population_fitness, current_population]

    with open('ga_information.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(generation_info)
