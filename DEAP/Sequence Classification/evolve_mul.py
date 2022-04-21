"""
This is an example of sequence classification using DEAP.

Example Input:
    sequence        = [1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0]
    Stack_output    = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0]
    
Example Output:
    Action_output   = [0.0, 2.0, 2.0, 0.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 0.0, 2.0] where 0=PUSH, 1=POP, 2=NONE
    Stack_output    = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0] where 0 means empty
    classification  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
"""

import itertools
import operator
import random
import numpy as np
import pickle
import multiprocessing
import os

from deap import gp
from deap import tools
from deap import base
from deap import creator
from deap import algorithms
from sklearn.metrics import accuracy_score

# Data Config
depth = 5              # Number of (1, -1) in a sequence
noise = 10              # Number of Zeros between values
num_tests = 50          # num_tests is the number of random examples each network is tested against.
num_runs = 50           # number of runs

# Results Config
generalize = True
save_log = True
verbose_val = False

# Directory of files
local_dir = os.path.dirname(__file__)
rpt_path = os.path.join(local_dir, 'reports/')
champ_path = os.path.join(local_dir, 'champions/')


# Generate Random Data
def generate_data(depth, noise):
    retval = []
    for _ in range(num_tests):
        sequence = []
        sequence.append(random.choice((-1.0, 1.0)))
        for _ in range(depth - 1):
            sequence.extend([0 for _ in range(noise)])
            sequence.append(random.choice((-1.0, 1.0)))
        retval.append(sequence)
    return retval

# Generate Classification based on dataset
def generate_output(dataset):
    retval = []
    for i in range(num_tests):
        data = dataset[i]
        sequence = []
        counter = 0
        for el in data:
            counter += el
            sequence.append(-1 if counter < 0 else 1)
        retval.append(sequence)
    return retval

# Generate expected GP Action based on Dataset
def generate_action(dataset):
    retval = []
    for i in range(num_tests):
        data = dataset[i]
        sequence = []
        MEMORY = []
        for el in data:
            if el == 0:
                sequence.append(2)
            else:
                if len(MEMORY) == 0 or MEMORY[len(MEMORY)-1] == el:
                    sequence.append(0)
                    MEMORY.append(el)
                else:
                    sequence.append(1)
                    MEMORY.pop()
        retval.append(sequence)
    return retval

# Generate Train Dataset
random_noise = noise

if generalize:
    random_noise = random.randint(10, 20)
data_train = generate_data(depth, random_noise)
labels_train = generate_output(data_train)
actions_train = generate_action(data_train)

'''
    Begining of DEAP Structure
'''

# Define a protected division function
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Calculate statistics
def calc_stats(output):
    record = {}

    # Calculate stats for first output
    output_avg = np.around(np.mean(output, axis=0), 2)
    output_std = np.around(np.std(output, axis=0), 2)
    output_min = np.around(np.min(output, axis=0), 2)
    output_max = np.around(np.max(output, axis=0), 2)
    output_length = len(output)

    # Update record
    record.update({
        'avg': output_avg,
        'std': output_std,
        'min': output_min,
        'max': output_max,
        'len': output_length,
    })

    return record


def eval_function(individual):
    # Transform the tree expression in a callable function
    tree1 = toolbox.compile(expr=individual[0])  # f1(x)
    tree2 = toolbox.compile(expr=individual[1])  # f2(x)
    tree3 = toolbox.compile(expr=individual[2])  # f3(x)
    tree4 = toolbox.compile(expr=individual[3])  # f4(x)

    fitness, total_len = 0, 0
    # Evaluate the sum of correctly identified
    for i in range(num_tests):
        data = data_train[i]
        labels = labels_train[i]
        actions = actions_train[i]
        MEMORY, classification = [], []
        counter = 0
        stopped = False
        length = len(data)
        total_len += length
        for j in range(length):
            # If stack is empty then 0, else the value on top of stack
            stack_output = MEMORY[counter - 1] if counter > 0 else 0

            arg1 = tree1(data[j],stack_output)
            arg2 = tree2(data[j],stack_output)
            arg3 = tree3(data[j],stack_output)
            arg4 = tree4(data[j],stack_output)
            pos = np.argmax([arg1, arg2, arg3, arg4])

            if pos == actions[j]:
                # correct action produced
                if pos == 0:
                    MEMORY.append(data[j])
                    temp = data[j]
                    counter += 1
                elif pos == 1:
                    MEMORY.pop()
                    counter -= 1
                    stack_output = MEMORY[counter - 1] if counter > 0 else 0
                    temp = 1 if stack_output >= 0 else -1
                else:
                    temp = 1 if stack_output >= 0 else -1
                
                # Add to classification
                if temp == labels[j]:
                    classification.append(temp)
                else:
                    print("Something has went horribly wrong!")
            else:
                # wrong action produced
                fitness += len(classification)
                stopped = True
                break
        if stopped == False:
            fitness += len(classification)

    return fitness/total_len,

# Champion Test for progress report
def champion_test(hof_array):
    tree1 = toolbox.compile(expr=hof_array[0])
    tree2 = toolbox.compile(expr=hof_array[1])
    tree3 = toolbox.compile(expr=hof_array[2])
    tree4 = toolbox.compile(expr=hof_array[3])

    # Generate Test Dataset
    random_noise = noise
    if generalize:
        random_noise = random.randint(10, 20)
    data_validation = generate_data(depth, random_noise)
    actions_validation = generate_action(data_validation)

    # Evaluate the sum of correctly identified
    predictions, predict_actions = [],[]
    total_accuracy = 0
    # Evaluate the sum of correctly identified
    for i in range(num_tests):
        data = data_validation[i]
        MEMORY, classification, actions = [], [], []
        counter = 0
        length = len(data)
        for j in range(length):
            # If stack is empty then 0, else the value on top of stack
            stack_output = MEMORY[counter - 1] if counter > 0 else 0

            arg1 = tree1(data[j],stack_output)
            arg2 = tree2(data[j],stack_output)
            arg3 = tree3(data[j],stack_output)
            arg4 = tree4(data[j],stack_output)
            pos = np.argmax([arg1, arg2, arg3, arg4])

            # Action has been decided
            actions.append(pos)
            if pos == 0:
                MEMORY.append(data[j])
                temp = data[j]
                counter += 1
            elif pos == 1:
                if len(MEMORY) > 0:
                    MEMORY.pop()
                counter -= 1
                stack_output = MEMORY[counter - 1] if counter > 0 else 0
                temp = 1 if stack_output >= 0 else -1
            else:
                temp = 1 if stack_output >= 0 else -1
            
            # Add to classification
            classification.append(temp)

        predictions.append(classification)
        predict_actions.append(actions)
        accuracy = accuracy_score(actions_validation[i], actions)
        if accuracy == 1.0:
            total_accuracy += 1
    
    progress_report.append((total_accuracy/num_tests)*100)

'''
Psudo code for how this algorithm works:
N = population size
population1 = [tree1, tree2, ..., treeN]
population2 = [tree1, tree2, ..., treeN]
population3 = [tree1, tree2, ..., treeN]

New trees have invalid fitness
invalid_ind1 = [invalid_tree1, invalid_tree2, ..., invalid_treeN]
invalid_ind2 = [invalid_tree1, invalid_tree2, ..., invalid_treeN]
invalid_ind3 = [invalid_tree1, invalid_tree2, ..., invalid_treeN]

for I = 1 to N
    evaluate_fitness(invalid_ind1[I], invalid_ind2[I], invalid_ind3[I])
Next I

Pick the best tree fitness and assign it to Hall Of Fame for each invalid_ind.

For each generation evolving include:
* Select the next generation individuals (select entire population):
    offspring1 = select(population1, len(population1))
    offspring2 = select(population2, len(population2))
    offspring3 = select(population3, len(population3))

* Vary the pool of individuals:
    offspring_population = clone(parent_population)
    For I = 1 to len(offspring_population)
        If mate_probability then
            child1, child2 = mate(offspring_population[I], offspring_population[I+1])
            offspring_population[I] = child1
            offspring_population[I+1] = child2
        End
    Next I
    For I = 1 to len(offspring_population)
        If mutate_probability then
            mutate_child = mutate(offspring_population[I])
            offspring_population[I] = mutate_childtemp_data
    
* Select new trees with invalid fittness then (number of invalid tree != N):
    for I = 1 to N
        evaluate_fitness(invalid_ind1[I], invalid_ind2[I], invalid_ind3[I])
    Next I
* Update the hall of fame with the generated individuals.
* Replace the current population with the offspring.
'''
def ea_simple_plus(population_list, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):

    population1 = population_list[0]
    population2 = population_list[1]
    population3 = population_list[2]
    population4 = population_list[3]

    halloffame1 = halloffame[0]
    halloffame2 = halloffame[1]
    halloffame3 = halloffame[2]
    halloffame4 = halloffame[3]

    # Evaluate the individuals with an invalid fitness
    invalid_ind1 = [ind for ind in population1 if not ind.fitness.valid]
    invalid_ind2 = [ind for ind in population2 if not ind.fitness.valid]
    invalid_ind3 = [ind for ind in population3 if not ind.fitness.valid]
    invalid_ind4 = [ind for ind in population4 if not ind.fitness.valid]

    # we need to zip the 4 individuals and pass it to the eval_function,
    # represented here as "toolbox.evaluate". The returned list of cost is then evaluated for each of the individuals.
    fitness = []
    fitnesses = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4))
    for ind1, ind2, ind3, ind4, fit in zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4, fitnesses):
        ind1.fitness.values = fit
        ind2.fitness.values = fit
        ind3.fitness.values = fit
        ind4.fitness.values = fit
        fitness.append(fit[0])

    if halloffame is not None:
        halloffame1.update(population1)
        halloffame2.update(population2)
        halloffame3.update(population3)
        halloffame4.update(population4)

    record = calc_stats(fitness)
    header = ['Gen'] + list(record.keys())
    values = [0] + list(record.values())

    # Test Champion and log it
    if save_log:
        hof_list = [halloffame1[0], halloffame2[0], halloffame3[0], halloffame4[0]]
        champion_test(hof_list)

    if verbose:
        print(*header, sep='\t')
        print(*values, sep='\t')
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring1 = toolbox.select(population1, len(population1))
        offspring2 = toolbox.select(population2, len(population2))
        offspring3 = toolbox.select(population3, len(population3))
        offspring4 = toolbox.select(population4, len(population4))

        # Vary the pool of individuals
        offspring1 = algorithms.varAnd(offspring1, toolbox, cxpb, mutpb)
        offspring2 = algorithms.varAnd(offspring2, toolbox, cxpb, mutpb)
        offspring3 = algorithms.varAnd(offspring3, toolbox, cxpb, mutpb)
        offspring4 = algorithms.varAnd(offspring4, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        fitness = []
        invalid_ind1 = [ind for ind in offspring1 if not ind.fitness.valid]
        invalid_ind2 = [ind for ind in offspring2 if not ind.fitness.valid]
        invalid_ind3 = [ind for ind in offspring3 if not ind.fitness.valid]
        invalid_ind4 = [ind for ind in offspring4 if not ind.fitness.valid]

        fitnesses = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4))
        for ind1, ind2, ind3, ind4, fit in zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4, fitnesses):
            ind1.fitness.values = fit
            ind2.fitness.values = fit
            ind3.fitness.values = fit
            ind4.fitness.values = fit
            fitness.append(fit)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame1.update(offspring1)
            halloffame2.update(offspring2)
            halloffame3.update(offspring3)
            halloffame4.update(offspring4)

        # Replace the current population with the offspring
        population1[:] = offspring1
        population2[:] = offspring2
        population3[:] = offspring3
        population4[:] = offspring4

        # Append the current generation statistics to the logbook
        record = calc_stats(fitness)
        values = [gen] + list(record.values())

        # Test Champion and log it
        if save_log:
            hof_list = [halloffame1[0], halloffame2[0], halloffame3[0], halloffame4[0]]
            champion_test(hof_list)

        if verbose:
            print(*values, sep='\t')

        if save_log == False and record['max'] >= fitness_threshold:
            break

    return [population1, population2, population3, population4]


# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 2), float)

# Float operators
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", eval_function)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Create 4 Individuals (4 outputs)
toolbox.register("individual1", tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("individual2", tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("individual3", tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("individual4", tools.initIterate,creator.Individual, toolbox.expr)

# Create output populations.
toolbox.register("population1", tools.initRepeat, list, toolbox.individual1)
toolbox.register("population2", tools.initRepeat, list, toolbox.individual2)
toolbox.register("population3", tools.initRepeat, list, toolbox.individual3)
toolbox.register("population4", tools.initRepeat, list, toolbox.individual4)

if __name__ == "__main__":
    champions, reports = {}, {}
    for i in range(num_runs):
        # Process Pool of ncpu workers
        ncpu = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=ncpu)
        toolbox.register("map", pool.map)
        progress_report = []

        pop_size = 100
        pop1 = toolbox.population1(n=pop_size)
        pop2 = toolbox.population2(n=pop_size)
        pop3 = toolbox.population3(n=pop_size)
        pop4 = toolbox.population4(n=pop_size)

        hof1 = tools.HallOfFame(1)
        hof2 = tools.HallOfFame(1)
        hof3 = tools.HallOfFame(1)
        hof4 = tools.HallOfFame(1)
        
        pop_list = [pop1, pop2, pop3, pop4]
        hof_list = [hof1, hof2, hof3, hof4]
        cxpb, mutpb, ngen, fitness_threshold = 0.5, 0.4, 250, 0.95

        if not verbose_val:
            print("Generation #: " + str(i+1))

        pop = ea_simple_plus(pop_list, toolbox, cxpb, mutpb, ngen, None, hof_list, verbose=verbose_val)

        if verbose_val:
            print("\nFirst Output Best individual fitness: %s" % (hof1[0].fitness))
            print("Second Output Best individual fitness: %s" % (hof2[0].fitness))
            print("Third Output Best individual fitness: %s" % (hof3[0].fitness))
            print("Fourth Output Best individual fitness: %s" % (hof4[0].fitness))

        # Save the winner
        champions["champion_" + str(i+1)] = [hof1[0], hof2[0], hof3[0], hof4[0]]
        reports['report' + str(i+1)] = progress_report

    # Save Champions
    with open(champ_path + str(depth) + '_champions_mul', 'wb') as f:
        pickle.dump(champions, f)

    if save_log:
        with open(rpt_path + str(depth) + '_report_mul', 'wb') as f:
            pickle.dump(reports, f)