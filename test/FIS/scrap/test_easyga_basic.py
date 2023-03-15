import EasyGA
import random

# Create the Genetic algorithm
ga = EasyGA.GA()

# Create 25 chromosomes each with 10 genes and 200 generations
ga.population_size = 25
ga.chromosome_length = 10
ga.generation_goal = 200

# Create random genes from 0 to 10
ga.gene_impl = lambda: random.randint(0, 10)

# Make the package minimize the fitness
ga.target_fitness_type = 'min'


def user_def_fitness(chromosome):
    """"The sum of the gene values. Take each gene value
     and add it to the chromosomes overall fitness."""

    fitness = 0

    for gene in chromosome:
        fitness += gene.value

    return fitness


ga.fitness_function_impl = user_def_fitness

ga.evolve()

# Print your default genetic algorithm
ga.print_generation()
ga.print_population()