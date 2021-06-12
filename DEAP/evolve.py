import json
import itertools
import operator
import random
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
def calc_stats(output1, output2, output3):
    record = {}

    # Calculate stats for first output
    output1_avg = np.around(np.mean(output1, axis=0), 2)
    output1_std = np.around(np.std(output1, axis=0), 2)
    output1_min = np.around(np.min(output1, axis=0), 2)
    output1_max = np.around(np.max(output1, axis=0), 2)

    # Calculate stats for second output
    output2_avg = np.around(np.mean(output2, axis=0), 2)
    output2_std = np.around(np.std(output2, axis=0), 2)
    output2_min = np.around(np.min(output2, axis=0), 2)
    output2_max = np.around(np.max(output2, axis=0), 2)

    # Calculate stats for third output
    output3_avg = np.around(np.mean(output3, axis=0), 2)
    output3_std = np.around(np.std(output3, axis=0), 2)
    output3_min = np.around(np.min(output3, axis=0), 2)
    output3_max = np.around(np.max(output3, axis=0), 2)

    # Update record
    record.update({
        'avg1': output1_avg,
        'std1': output1_std,
        'min1': output1_min,
        'max1': output1_max,
        'avg2': output2_avg,
        'std2': output2_std,
        'min2': output2_min,
        'max2': output2_max,
        'avg3': output3_avg,
        'std3': output3_std,
        'min3': output3_min,
        'max3': output3_max,
    })

    return record


def eval_function(individual):
    # Transform the tree expression in a callable function
    tree1 = toolbox.compile(expr=individual[0])  # f1(x)
    tree2 = toolbox.compile(expr=individual[1])  # f2(x)
    tree3 = toolbox.compile(expr=individual[2])  # f3(x)

    # Evaluate the sum of correctly identified
    total = []
    for i in range(len(data_train)):
        arg1 = tree1(*data_train[i])
        arg2 = tree2(*data_train[i])
        arg3 = tree3(*data_train[i])
        pos = np.argmax([arg1, arg2, arg3])
        total.append((pos == labels_train[i]))

    return sum(total)/len(data_train),


def ea_simple_plus(population_list, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):

    population1 = population_list[0]
    population2 = population_list[1]
    population3 = population_list[2]

    halloffame1 = halloffame[0]
    halloffame2 = halloffame[1]
    halloffame3 = halloffame[2]

    # Evaluate the individuals with an invalid fitness
    invalid_ind1 = [ind for ind in population1 if not ind.fitness.valid]
    invalid_ind2 = [ind for ind in population2 if not ind.fitness.valid]
    invalid_ind3 = [ind for ind in population3 if not ind.fitness.valid]

    # we need to zip the 3 individuals and pass it to the eval_function,
    # represented here as "toolbox.evaluate". The returned list of cost is then evaluated for each of the individuals.
    fitness1 = []
    fitnesses1 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
    for ind, fit in zip(invalid_ind1, fitnesses1):
        ind.fitness.values = fit
        fitness1.append(fit[0])

    fitness2 = []
    fitnesses2 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
    for ind, fit in zip(invalid_ind2, fitnesses2):
        ind.fitness.values = fit
        fitness2.append(fit[0])

    fitness3 = []
    fitnesses3 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
    for ind, fit in zip(invalid_ind3, fitnesses3):
        ind.fitness.values = fit
        fitness3.append(fit[0])

    if halloffame is not None:
        halloffame1.update(population1)
        halloffame2.update(population2)
        halloffame3.update(population3)

    record = calc_stats(fitness1, fitness2, fitness3)
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
        offspring3 = toolbox.select(population3, len(population3))

        # Vary the pool of individuals
        offspring1 = algorithms.varAnd(offspring1, toolbox, cxpb, mutpb)
        offspring2 = algorithms.varAnd(offspring2, toolbox, cxpb, mutpb)
        offspring3 = algorithms.varAnd(offspring3, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        fitness1 = []
        invalid_ind1 = [ind for ind in offspring1 if not ind.fitness.valid]
        fitnesses1 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
        for ind, fit in zip(invalid_ind1, fitnesses1):
            ind.fitness.values = fit
            fitness1.append(fit)

        fitness2 = []
        invalid_ind2 = [ind for ind in offspring2 if not ind.fitness.valid]
        fitnesses2 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
        for ind, fit in zip(invalid_ind2, fitnesses2):
            ind.fitness.values = fit
            fitness2.append(fit)

        fitness3 = []
        invalid_ind3 = [ind for ind in offspring3 if not ind.fitness.valid]
        fitnesses3 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
        for ind, fit in zip(invalid_ind3, fitnesses3):
            ind.fitness.values = fit
            fitness3.append(fit)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame1.update(offspring1)
            halloffame2.update(offspring2)
            halloffame3.update(offspring3)

        # Replace the current population by the offspring
        population1[:] = offspring1
        population2[:] = offspring2
        population3[:] = offspring3

        # Append the current generation statistics to the logbook
        record = calc_stats(fitness1, fitness2, fitness3)
        values = [gen] + list(record.values())

        if verbose:
            print(*values, sep='\t')

        if record['max1'] > fitness_threshold and record['max2'] > fitness_threshold and record['max3'] > fitness_threshold:
            break

    return [population1, population2, population3]


# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 4), float, "IN")

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
toolbox.register("individual3", tools.initIterate,creator.Individual, toolbox.expr)

# Create output populations.
toolbox.register("population1", tools.initRepeat, list, toolbox.individual1)
toolbox.register("population2", tools.initRepeat, list, toolbox.individual2)
toolbox.register("population3", tools.initRepeat, list, toolbox.individual3)

pop_number = 100
pop1 = toolbox.population1(n=pop_number)
pop2 = toolbox.population2(n=pop_number)
pop3 = toolbox.population3(n=pop_number)

hof1 = tools.HallOfFame(1)
hof2 = tools.HallOfFame(1)
hof3 = tools.HallOfFame(1)


if __name__ == "__main__":

    # Loading the Data Set
    with open("iris.json") as file:
        iris = json.load(file)

    # Split-out validation dataset
    data = iris['data']
    labels = iris['target']
    data_train, data_validation, labels_train, labels_validation = train_test_split(
        data, labels, test_size=0.20, random_state=1)

    pop_list = [pop1, pop2, pop3]
    hof_list = [hof1, hof2, hof3]
    cxpb, mutpb, ngen, fitness_threshold = 0.5, 0.4, 100, 0.95
    pop = ea_simple_plus(pop_list, toolbox, cxpb, mutpb, ngen, None, hof_list, verbose=True)

    print("\nFirst Output Best individual fitness: %s" % (hof1[0].fitness))
    print("Second Output Best individual fitness: %s" % (hof2[0].fitness))
    print("Third Output Best individual fitness: %s" % (hof3[0].fitness))

    # Save the winner
    with open('output1', 'wb') as f:
        pickle.dump(hof1[0], f)

    with open('output2', 'wb') as f:
        pickle.dump(hof2[0], f)

    with open('output3', 'wb') as f:
        pickle.dump(hof3[0], f)

    '''
    Running Test on unseen data and checking results
    '''

    print("\n==================")
    print("Begin Testing ....")
    print("==================\n")
    # Transform the tree expression in a callable function
    tree1 = toolbox.compile(expr=hof1[0])
    tree2 = toolbox.compile(expr=hof2[0])
    tree3 = toolbox.compile(expr=hof3[0])

    # Evaluate the sum of correctly identified
    predictions = []
    for i in range(len(data_validation)):
        arg1 = tree1(*data_validation[i])
        arg2 = tree2(*data_validation[i])
        arg3 = tree3(*data_validation[i])
        pos = np.argmax([arg1, arg2, arg3])
        # predictions.append((pos == labels_validation[i]))
        predictions.append(pos)

    # Evaluate predictions
    accuracy = accuracy_score(labels_validation, predictions)
    print("Accuracy: {}".format(accuracy))
    print(classification_report(labels_validation, predictions))
    print("Predictions: \n{}".format(predictions))
    print("labels: \n{}".format(labels_validation))
