"""
This is an example of Copy Task using DEAP (1-bit).

Example Input:
    for sequence length of 3, Write delim is in first position, Read delim is in second position.
    Sample Input = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
Example Output:
    Abstracted output, if DEAP guess the command right the stack will be correct. This is why we only care about commands.
    Sample Output = [[2, 0, 0, 0, 2, 1, 1, 1]]
"""

import itertools
import operator
import random
import numpy as np
import pickle
import multiprocessing

from deap import gp
from deap import tools
from deap import base
from deap import creator
from deap import algorithms
from sklearn.metrics import accuracy_score

# length of the test sequence.
seq_length = 10
# number of bits used
bits = 8
# num_tests is the number of random examples each network is tested against.
num_tests = 50
generalize = False
save_log = False

'''
Problem setup
'''

def generate_data(seq_length):
    retval = []
    for _ in range(num_tests):
        if generalize:
                seq_length = random.randint(10, 20)
        # Adding 2 to bits for writing delim and reading delim
        # also adding 2 to length for delim sequence
        sequence = np.zeros([seq_length + 2, bits + 2], dtype=np.float32)
        for idx in range(1, seq_length + 1):
            sequence[idx, 2:bits+2] = np.random.rand(bits).round()

        sequence[0, 0] = 1                # Setting Wrting delim
        sequence[seq_length+1, 1] = 1     # Setting reading delim

        recall = np.zeros([seq_length, bits + 2], dtype=np.float32)
        data = np.concatenate((sequence, recall), axis=0).tolist()
        retval.append(data)
    return retval

def generate_action(data_array):
    retval = []
    for i in range(num_tests):
        data, action, write, read = data_array[i], [], False, False
        length = len(data)

        # 0 = PUSH, 1 = POP HEAD, 2 = NOTHING, 3 = POP TAIL
        for x in range(length):
            if data[x][0] == 1 and data[x][1] == 0:
                write = True
                read = False
                action.append(2)
            elif data[x][0] == 0 and data[x][1] == 1:
                write = False
                read = True
                action.append(2)
            else:
                if write == True:
                    action.append(0)
                elif read == True:
                    action.append(1)
        retval.append(action)
    return retval
        

data_train = generate_data(seq_length)
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
        data, actions = data_train[i], actions_train[i]
        length = len(data)
        total_len += length

        for j in range(length):
            arg1 = tree1(data[j][0], data[j][1])
            arg2 = tree2(data[j][0], data[j][1])
            arg3 = tree3(data[j][0], data[j][1])
            arg4 = tree4(data[j][0], data[j][1])
            pos = np.argmax([arg1, arg2, arg3, arg4])

            if pos == actions[j]:
                fitness += 1
            else:
                # wrong action produced
                break
    return fitness/total_len,

# Champion Test for progress report
def champion_test(hof_array):
    tree1 = toolbox.compile(expr=hof_array[0])
    tree2 = toolbox.compile(expr=hof_array[1])
    tree3 = toolbox.compile(expr=hof_array[2])
    tree4 = toolbox.compile(expr=hof_array[3])

    # Generate Test Dataset
    data_validation = generate_data(seq_length)
    actions_validation = generate_action(data_validation)

    # Evaluate the sum of correctly identified
    predict_actions = []
    total_accuracy = 0
    # Evaluate the sum of correctly identified
    for i in range(num_tests):
        data, actions = data_validation[i], []
        length = len(data)

        for j in range(length):
            arg1 = tree1(data[j][0], data[j][1])
            arg2 = tree2(data[j][0], data[j][1])
            arg3 = tree3(data[j][0], data[j][1])
            arg4 = tree4(data[j][0], data[j][1])
            pos = np.argmax([arg1, arg2, arg3, arg4])
            actions.append(pos)

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
population4 = [tree1, tree2, ..., treeN]

New trees have invalid fitness
invalid_ind1 = [invalid_tree1, invalid_tree2, ..., invalid_treeN]
invalid_ind2 = [invalid_tree1, invalid_tree2, ..., invalid_treeN]
invalid_ind3 = [invalid_tree1, invalid_tree2, ..., invalid_treeN]
invalid_ind4 = [invalid_tree1, invalid_tree2, ..., invalid_treeN]

for I = 1 to N
    evaluate_fitness(invalid_ind1[I], invalid_ind2[I], invalid_ind3[I], invalid_ind4[I])
Next I

Pick the best tree fitness and assign it to Hall Of Fame for each invalid_ind.

For each generation evolving include:
* Select the next generation individuals (select entire population):
    offspring1 = select(population1, len(population1))
    offspring2 = select(population2, len(population2))
    offspring3 = select(population3, len(population3))
    offspring4 = select(population3, len(population4))

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
        evaluate_fitness(invalid_ind1[I], invalid_ind2[I], invalid_ind3[I], invalid_ind4[I])
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

    # we need to zip the 3 individuals and pass it to the eval_function,
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

        if record['max'] >= fitness_threshold:
            break

    return [population1, population2, population3, population4]


# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 2), float)

# Float operators
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(protected_div, [float, float], float)

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

# Create 3 Individuals (3 outputs)
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
    # for i in range(1, 11):
    # Process Pool of ncpu workers
    ncpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=ncpu)
    toolbox.register("map", pool.map)
    progress_report = []

    pop_size = 100
    pop1 = toolbox.population1(n=pop_size)
    pop2 = toolbox.population2(n=pop_size)
    pop3 = toolbox.population2(n=pop_size)
    pop4 = toolbox.population2(n=pop_size)

    hof1 = tools.HallOfFame(1)
    hof2 = tools.HallOfFame(1)
    hof3 = tools.HallOfFame(1)
    hof4 = tools.HallOfFame(1)
    
    pop_list = [pop1, pop2, pop3, pop4]
    hof_list = [hof1, hof2, hof3, hof4]
    cxpb, mutpb, ngen, fitness_threshold = 0.5, 0.4, 100, 0.95
    pop = ea_simple_plus(pop_list, toolbox, cxpb, mutpb, ngen, None, hof_list, verbose=True)

    print("\nFirst Output Best individual fitness: %s" % (hof1[0].fitness))
    print("Second Output Best individual fitness: %s" % (hof2[0].fitness))
    print("Third Output Best individual fitness: %s" % (hof3[0].fitness))
    print("Fourth Output Best individual fitness: %s" % (hof4[0].fitness))

    # Save the winner
    with open('output1', 'wb') as f:
        pickle.dump(hof1[0], f)

    with open('output2', 'wb') as f:
        pickle.dump(hof2[0], f)
    
    with open('output3', 'wb') as f:
        pickle.dump(hof3[0], f)

    with open('output4', 'wb') as f:
        pickle.dump(hof4[0], f)

    if save_log:
        with open(str(seq_length) + '-progress_report' + str(i), 'wb') as f:
            pickle.dump(progress_report, f)