import json
import itertools
import operator
import random
import numpy

from sklearn.model_selection import train_test_split
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
    result1 = sum(tree1(*indata) == labels_train for indata in data_train)
    result2 = sum(tree2(*indata) == labels_train for indata in data_train)
    result3 = sum(tree3(*indata) == labels_train for indata in data_train)
    total = result1 + result2 + result3

    return total,


def evolve(pop_list, toolbox, cxpb, mutpb, ngen, stats=None, hof_list=None, verbose=__debug__):

    pop1 = pop_list[0]
    pop2 = pop_list[1]
    pop3 = pop_list[2]

    hof1 = hof_list[0]
    hof2 = hof_list[1]
    hof3 = hof_list[2]

    logbook1 = tools.Logbook()
    logbook2 = tools.Logbook()
    logbook3 = tools.Logbook()

    # Evaluate the individuals with an invalid fitness
    invalid_ind1 = [ind for ind in pop1 if not ind.fitness.valid]
    invalid_ind2 = [ind for ind in pop2 if not ind.fitness.valid]
    invalid_ind3 = [ind for ind in pop3 if not ind.fitness.valid]
    
    # we need to zip the 3 individuals and pass it to the eval_function,
    # represented here as "toolbox.evaluate". The returned list of cost is then evaluated for each of the individuals.
    fitnesse1 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
    for ind, fit in zip(invalid_ind1, fitnesse1):
        ind.fitness.values = fit

    fitnesse2 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
    for ind, fit in zip(invalid_ind2, fitnesse2):
        ind.fitness.values = fit

    fitnesse3 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
    for ind, fit in zip(invalid_ind3, fitnesse3):
        ind.fitness.values = fit

    hof1.update(pop1)
    hof2.update(pop2)
    hof3.update(pop3)

    record1 = stats.compile(pop1) if stats else {}
    record2 = stats.compile(pop2) if stats else {}
    record3 = stats.compile(pop3) if stats else {}

    logbook1.record(gen=0, no_of_evals=len(invalid_ind1), **record1)
    logbook2.record(gen=0, no_of_evals=len(invalid_ind2), **record2)
    logbook3.record(gen=0, no_of_evals=len(invalid_ind3), **record3)

    if verbose:
        print(logbook1.stream, logbook2.stream, logbook3.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring1 = toolbox.select(pop1, len(pop1))
        offspring2 = toolbox.select(pop2, len(pop2))
        offspring3 = toolbox.select(pop3, len(pop3))

        # Vary the pool of individuals
        offspring1 = algorithms.varAnd(offspring1, toolbox, cxpb, mutpb)
        offspring2 = algorithms.varAnd(offspring2, toolbox, cxpb, mutpb)
        offspring3 = algorithms.varAnd(offspring3, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind1 = [ind for ind in offspring1 if not ind.fitness.valid]
        invalid_ind2 = [ind for ind in offspring2 if not ind.fitness.valid]
        invalid_ind3 = [ind for ind in offspring3 if not ind.fitness.valid]

        fitness1 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
        for ind, fit in zip(invalid_ind1, fitness1):
            ind.fitness.values = fit

        fitness2 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
        for ind, fit in zip(invalid_ind2, fitness2):
            ind.fitness.values = fit

        fitness3 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3))
        for ind, fit in zip(invalid_ind3, fitness3):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        hof1.update(offspring1)
        hof2.update(offspring2)
        hof3.update(offspring3)

        # Replace the current population by the offspring
        pop1[:] = offspring1
        pop2[:] = offspring2
        pop3[:] = offspring3

        # Append the current generation statistics to the logbook
        record1 = stats.compile(pop1) if stats else {}
        record2 = stats.compile(pop2) if stats else {}
        record3 = stats.compile(pop3) if stats else {}

        test = stats.compile(pop1)
        print('Record: {}'.format(test))
        
        logbook1.record(gen=gen, no_of_evals=len(invalid_ind1), **record1)
        logbook2.record(gen=gen, no_of_evals=len(invalid_ind2), **record2)
        logbook3.record(gen=gen, no_of_evals=len(invalid_ind3), **record3)

        if verbose:
            print(logbook1.stream, logbook2.stream, logbook3.stream)

    pop = [pop1, pop2, pop3]
    log = [logbook1, logbook2, logbook3]
    return pop, log


# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 4), float, "IN")

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
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
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=5, type_=bool)
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

pop1 = toolbox.population1(n=300)
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
    pop_list = [pop1, pop2, pop3]
    hof_list = [hof1, hof2, hof3]
    cxpb = 0.5
    mutpb = 0.4
    ngen = 40
    pop, log = evolve(pop_list, toolbox, cxpb, mutpb, ngen, stats=stats, hof_list=hof_list, verbose=True)
    