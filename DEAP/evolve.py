import json
import itertools
import operator
import random
import numpy
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
        pos = numpy.argmax([arg1, arg2, arg3])
        total.append((pos == labels_train[i])) 

    return sum(total),

def ea_simple_plus(population_list, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    
    logbook1 = tools.Logbook()
    logbook2 = tools.Logbook()
    logbook3 = tools.Logbook()

    logbook1.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    logbook2.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    logbook3.header = ['gen', 'nevals'] + (stats.fields if stats else [])

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
    fitnesses1 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
    for ind, fit in zip(invalid_ind1, fitnesses1):
        ind.fitness.values = fit

    fitnesses2 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
    for ind, fit in zip(invalid_ind2, fitnesses2):
        ind.fitness.values = fit

    fitnesses3 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
    for ind, fit in zip(invalid_ind3, fitnesses3):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame1.update(population1)
        halloffame2.update(population2)
        halloffame3.update(population3)

    record1 = stats.compile(population1) if stats else {}
    logbook1.record(gen=0, nevals=len(invalid_ind1), **record1)

    if verbose:
        print(logbook1.stream)

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
        invalid_ind1 = [ind for ind in offspring1 if not ind.fitness.valid]
        fitnesses1 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
        for ind, fit in zip(invalid_ind1, fitnesses1):
            ind.fitness.values = fit

        invalid_ind2 = [ind for ind in offspring2 if not ind.fitness.valid]
        fitnesses2 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
        for ind, fit in zip(invalid_ind2, fitnesses2):
            ind.fitness.values = fit

        invalid_ind3 = [ind for ind in offspring3 if not ind.fitness.valid]
        fitnesses3 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
        for ind, fit in zip(invalid_ind3, fitnesses3):
            ind.fitness.values = fit

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
        record1 = stats.compile(population1) if stats else {}
        logbook1.record(gen=gen, nevals=len(invalid_ind1), **record1)

        if verbose:
            print(logbook1.stream)

    return [population1, population2, population3], [logbook1, logbook2, logbook3]


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

pop1 = toolbox.population1(n=10)
pop2 = toolbox.population2(n=300)
pop3 = toolbox.population3(n=300)

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

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    pop_list = [pop1, pop2, pop3]
    hof_list = [hof1, hof2, hof3]
    cxpb, mutpb, ngen = 0.5, 0.4, 40
    pop, log = ea_simple_plus(pop_list, toolbox, cxpb, mutpb, ngen, stats, hof_list, verbose=True)

    print("First Output Best individual fitness: %s" % (hof1[0].fitness))
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
        pos = numpy.argmax([arg1, arg2, arg3])
        # predictions.append((pos == labels_validation[i]))
        predictions.append(pos)

    # Evaluate predictions
    accuracy = accuracy_score(labels_validation, predictions)
    print("Accuracy: {}".format(accuracy))
    print(classification_report(labels_validation, predictions))
    print("Predictions: \n{}".format(predictions))
    print("labels: \n{}".format(labels_validation))