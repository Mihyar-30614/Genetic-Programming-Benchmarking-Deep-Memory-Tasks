import itertools
import operator
import random
import numpy as np
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from deap import gp
from deap import tools
from deap import base
from deap import creator
from deap import algorithms

# Define a protected division function
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2

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

    # Evaluate the sum of correctly identified
    total = []
    for i in range(len(data_train)):
        arg1 = tree1(*data_train[i])
        arg2 = tree2(*data_train[i])
        pos = np.argmax([arg1, arg2])
        total.append((pos == labels_train[i]))

    return sum(total)/len(data_train),

'''
Psudo code for how this algorithm works:
N = population size
population1 = [tree1, tree2, ..., treeN]
population2 = [tree1, tree2, ..., treeN]

New trees have invalid fitness
invalid_ind1 = [invalid_tree1, invalid_tree2, ..., invalid_treeN]
invalid_ind2 = [invalid_tree1, invalid_tree2, ..., invalid_treeN]

for I = 1 to N
    evaluate_fitness(invalid_ind1[I], invalid_ind2[I])
Next I

Pick the best tree fitness and assign it to Hall Of Fame for each invalid_ind.

For each generation evolving include:
* Select the next generation individuals (select entire population):
    offspring1 = select(population1, len(population1))
    offspring2 = select(population2, len(population2))

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
            offspring_population[I] = mutate_child
        End
    Next I
    return offspring_population
    
* Select new trees with invalid fittness then (number of invalid tree != N):
    for I = 1 to N
        evaluate_fitness(invalid_ind1[I], invalid_ind2[I])
    Next I
* Update the hall of fame with the generated individuals.
* Replace the current population with the offspring.
'''
def ea_simple_plus(population_list, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):

    population1 = population_list[0]
    population2 = population_list[1]

    halloffame1 = halloffame[0]
    halloffame2 = halloffame[1]

    # Evaluate the individuals with an invalid fitness
    invalid_ind1 = [ind for ind in population1 if not ind.fitness.valid]
    invalid_ind2 = [ind for ind in population2 if not ind.fitness.valid]

    # we need to zip the 3 individuals and pass it to the eval_function,
    # represented here as "toolbox.evaluate". The returned list of cost is then evaluated for each of the individuals.
    fitness = []
    fitnesses = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2))
    for ind1, ind2, fit in zip(invalid_ind1,invalid_ind2, fitnesses):
        ind1.fitness.values = fit
        ind2.fitness.values = fit
        fitness.append(fit[0])

    if halloffame is not None:
        halloffame1.update(population1)
        halloffame2.update(population2)

    record = calc_stats(fitness)
    header = ['Gen'] + list(record.keys())
    values = [0] + list(record.values())

    if verbose:
        print(*header, sep='\t')
        print(*values, sep='\t')
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring1 = toolbox.select(population1, len(population1))
        offspring2 = toolbox.select(population2, len(population2))

        # Vary the pool of individuals
        offspring1 = algorithms.varAnd(offspring1, toolbox, cxpb, mutpb)
        offspring2 = algorithms.varAnd(offspring2, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        fitness = []
        invalid_ind1 = [ind for ind in offspring1 if not ind.fitness.valid]
        invalid_ind2 = [ind for ind in offspring2 if not ind.fitness.valid]

        fitnesses = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2))
        for ind1, ind2, fit in zip(invalid_ind1, invalid_ind2, fitnesses):
            ind1.fitness.values = fit
            ind2.fitness.values = fit
            fitness.append(fit)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame1.update(offspring1)
            halloffame2.update(offspring2)

        # Replace the current population with the offspring
        population1[:] = offspring1
        population2[:] = offspring2

        # Append the current generation statistics to the logbook
        record = calc_stats(fitness)
        values = [gen] + list(record.values())

        if verbose:
            print(*values, sep='\t')

        if record['max'] >= fitness_threshold:
            break

    return [population1, population2]


# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 9), float)

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(protected_div, [float, float], float)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=5)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", eval_function)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=3, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Create 3 Individuals (3 outputs)
toolbox.register("individual1", tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("individual2", tools.initIterate,creator.Individual, toolbox.expr)

# Create output populations.
toolbox.register("population1", tools.initRepeat, list, toolbox.individual1)
toolbox.register("population2", tools.initRepeat, list, toolbox.individual2)

pop_size = 100
pop1 = toolbox.population1(n=pop_size)
pop2 = toolbox.population2(n=pop_size)

hof1 = tools.HallOfFame(1)
hof2 = tools.HallOfFame(1)


if __name__ == "__main__":

    # Loading the Train Dataset
    lines = np.loadtxt("Train-dataset.txt", comments="#", delimiter=",", unpack=False, dtype=float)
    data_train = lines[:,0:9]
    labels_train = lines[:,9:].flatten()

    # Loading the Test Dataset
    lines = np.loadtxt("Test-dataset.txt", comments="#", delimiter=",", unpack=False, dtype=float)
    data_validation = lines[:,0:9]
    labels_validation = lines[:,9:].flatten()

    pop_list = [pop1, pop2]
    hof_list = [hof1, hof2]
    cxpb, mutpb, ngen, fitness_threshold = 0.5, 0.4, 40, 0.70
    pop = ea_simple_plus(pop_list, toolbox, cxpb, mutpb, ngen, None, hof_list, verbose=True)

    print("\nFirst Output Best individual fitness: %s" % (hof1[0].fitness))
    print("Second Output Best individual fitness: %s" % (hof2[0].fitness))

    # Save the winner
    with open('output1', 'wb') as f:
        pickle.dump(hof1[0], f)

    with open('output2', 'wb') as f:
        pickle.dump(hof2[0], f)

    '''
    Running Test on unseen data and checking results
    '''

    print("\n==================")
    print("Begin Testing ....")
    print("==================\n")
    # Transform the tree expression in a callable function
    tree1 = toolbox.compile(expr=hof1[0])
    tree2 = toolbox.compile(expr=hof2[0])

    # Evaluate the sum of correctly identified
    predictions = []
    for i in range(len(data_validation)):
        arg1 = tree1(*data_validation[i])
        arg2 = tree2(*data_validation[i])
        pos = np.argmax([arg1, arg2])
        predictions.append(pos)

    # Evaluate predictions
    accuracy = accuracy_score(labels_validation, predictions)
    print("Accuracy: {}".format(accuracy))
    print(classification_report(labels_validation, predictions))
    print("Predictions: \n{}".format(predictions))
    print("labels: \n{}".format(labels_validation))