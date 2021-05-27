
'''
    Resources to look at:
    https://colab.research.google.com/github/mariosky/databook/blob/master/DataLoad.ipynb#scrollTo=UrZylYdnwRRZ
    https://github.com/DEAP/deap/blob/master/examples/gp/multiplexer.py

    https://github.com/DEAP/deap/issues/491

    I have a 6-input 2-output problem. Basically, I need to find a function that would map the 6-inputs to the 2-outputs, hence minimizing the error. 
    (Just like symbolic regression, but with multiple output). Here are the following changes that I could do so far:
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,-1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    Now, I need to use the gp.compile tool to create a function func that would return two values of the cost function.
    Could you please guide me on how to do this? I saw several examples on multi-objectives, but none of them had the symbolic function involved.
    Thank you in advance. 

    Right now a tree can can return just a single value. A work around would be to create a bag of two trees and optimize the output of the bag.

    Right. Like you said, I have created a list of populations, and evolved it based on a single cost function. 
    This way, just by returning a single cost (thus, a single maximization problem), all the population members in the list is evolved.
    Here is the code snippet that I have used, if it can be of any help to others!
    In the following problem, I have 10 input variables, and 4 output variables. 
    Based on a cost function, all the 4 outputs must be maximized. I have rewritten the eaSimple(.) function for this purpose. 
    Please change it according to your requirements.
'''

import operator
import math
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define the primitives, and the required operators. I have re-written the division operator (as seen in symbreg.py example)
# There are 10 inputs. Further, if each of these 10 inputs are of different types, you can use the gp.PrimitiveSetTyped(.) function
pset = gp.PrimitiveSet("MAIN", 10)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(Div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.tanh, 1)
pset.addEphemeralConstant("rand01", lambda: random.randint(0, 1))

# Create only a single maximization problem, based on which each individual is assessed.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=5, max_=10)

# Create 4 individuals (as there are 4 outputs), and define their populations.
toolbox.register("individual1", tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("individual2", tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("individual3", tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("individual4", tools.initIterate,creator.Individual, toolbox.expr)

toolbox.register("population1", tools.initRepeat, list, toolbox.individual1)
toolbox.register("population2", tools.initRepeat, list, toolbox.individual2)
toolbox.register("population3", tools.initRepeat, list, toolbox.individual3)
toolbox.register("population4", tools.initRepeat, list, toolbox.individual4)
toolbox.register("compile", gp.compile, pset=pset)


# Define your cost function here
def cost_function(individual):
    func1 = toolbox.compile(expr=individual[0])  # f1(x)
    func2 = toolbox.compile(expr=individual[1])  # f2(x)
    func3 = toolbox.compile(expr=individual[2])  # f3(x)
    func4 = toolbox.compile(expr=individual[3])  # f4(x)
    # where x is a list containing the inputs. In the present example, this list would have 10 elements, as there are 10 inputs.

    # write what you want to do with these functions, and hence calculate the required cost.
    # For eg., cost = math.fsum((func1(a)* for a in x) + (func2(a)* for a in x) + (func3(a)* for a in x) + (func4(a)* for a in x))

    return cost,


# Re-write the eaSimple() function to evolve the 4 individuals w.r.t to the cost returned by the python function: cost_function
def Evolve(pop, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    pop1 = pop[0]
    pop2 = pop[1]
    pop3 = pop[2]
    pop4 = pop[3]

    hof1 = halloffame[0]
    hof2 = halloffame[1]
    hof3 = halloffame[2]
    hof4 = halloffame[3]

    logbook1 = tools.Logbook()
    logbook2 = tools.Logbook()
    logbook3 = tools.Logbook()
    logbook4 = tools.Logbook()

    logbook1.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    logbook2.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    logbook3.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    logbook4.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind1 = [ind for ind in pop1 if not ind.fitness.valid]
    invalid_ind2 = [ind for ind in pop2 if not ind.fitness.valid]
    invalid_ind3 = [ind for ind in pop3 if not ind.fitness.valid]
    invalid_ind4 = [ind for ind in pop4 if not ind.fitness.valid]

    # Here, the function cost_function takes in 4 inputs. Thus, we need to zip the 4 individuals and pass it to the cost_function, that is represented here as "toolbox.evaluate". The returned list of cost is then evaluated for each of the individuals.
    fitnesses1 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4))
    for ind, fit in zip(invalid_ind1, fitnesses1):
        ind.fitness.values = fit

    fitnesses2 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4))
    for ind, fit in zip(invalid_ind2, fitnesses2):
        ind.fitness.values = fit

    fitnesses3 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4))
    for ind, fit in zip(invalid_ind3, fitnesses3):
        ind.fitness.values = fit

    fitnesses4 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4))
    for ind, fit in zip(invalid_ind4, fitnesses4):
        ind.fitness.values = fit

    hof1.update(pop1)
    hof2.update(pop2)
    hof3.update(pop3)
    hof4.update(pop4)

    record1 = stats.compile(pop1) if stats else {}
    record2 = stats.compile(pop2) if stats else {}
    record3 = stats.compile(pop3) if stats else {}
    record4 = stats.compile(pop4) if stats else {}

    logbook1.record(gen=0, nevals=len(invalid_ind1), **record1)
    logbook2.record(gen=0, nevals=len(invalid_ind2), **record2)
    logbook3.record(gen=0, nevals=len(invalid_ind3), **record3)
    logbook4.record(gen=0, nevals=len(invalid_ind4), **record4)

    if verbose:
        print(logbook1.stream, logbook2.stream,
              logbook3.stream, logbook4.stream)

    for gen in range(1, ngen + 1):
        offspring1 = toolbox.select(pop1, len(pop1))
        offspring2 = toolbox.select(pop2, len(pop2))
        offspring3 = toolbox.select(pop3, len(pop3))
        offspring4 = toolbox.select(pop4, len(pop4))

        offspring1 = algorithms.varAnd(offspring1, toolbox, cxpb, mutpb)
        offspring2 = algorithms.varAnd(offspring2, toolbox, cxpb, mutpb)
        offspring3 = algorithms.varAnd(offspring3, toolbox, cxpb, mutpb)
        offspring4 = algorithms.varAnd(offspring4, toolbox, cxpb, mutpb)

        invalid_ind1 = [ind for ind in offspring1 if not ind.fitness.valid]
        invalid_ind2 = [ind for ind in offspring2 if not ind.fitness.valid]
        invalid_ind3 = [ind for ind in offspring3 if not ind.fitness.valid]
        invalid_ind4 = [ind for ind in offspring4 if not ind.fitness.valid]

        fitness1 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4))
        for ind, fit in zip(invalid_ind1, fitness1):
            ind.fitness.values = fit

        fitness2 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4))
        for ind, fit in zip(invalid_ind2, fitness2):
            ind.fitness.values = fit

        fitness3 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4))
        for ind, fit in zip(invalid_ind3, fitness3):
            ind.fitness.values = fit

        fitness4 = toolbox.map(toolbox.evaluate, zip(invalid_ind1, invalid_ind2, invalid_ind3, invalid_ind4))
        for ind, fit in zip(invalid_ind4, fitness4):
            ind.fitness.values = fit

        hof1.update(offspring1)
        hof2.update(offspring2)
        hof3.update(offspring3)
        hof4.update(offspring4)

        pop1[:] = offspring1
        pop2[:] = offspring2
        pop3[:] = offspring3
        pop4[:] = offspring4

        record1 = stats.compile(pop1) if stats else {}
        record2 = stats.compile(pop2) if stats else {}
        record3 = stats.compile(pop3) if stats else {}
        record4 = stats.compile(pop4) if stats else {}

        logbook1.record(gen=gen, nevals=len(invalid_ind1), **record1)
        logbook2.record(gen=gen, nevals=len(invalid_ind2), **record2)
        logbook3.record(gen=gen, nevals=len(invalid_ind3), **record3)
        logbook4.record(gen=gen, nevals=len(invalid_ind4), **record4)

        if verbose:
            print(logbook1.stream, logbook2.stream,
                  logbook3.stream, logbook4.stream)

    pop = [pop1, pop2, pop3, pop4]
    log = [logbook1, logbook2, logbook3, logbook4]
    return pop, log


# Define the parameters for the genetic programming.
# Thus, the function cost_function is now represented by "toolbox.evaluate"
toolbox.register("evaluate", cost_function)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=3, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

pop1 = toolbox.population1(n=300)
pop2 = toolbox.population2(n=300)
pop3 = toolbox.population1(n=300)
pop4 = toolbox.population2(n=300)

hof1 = tools.HallOfFame(1)
hof2 = tools.HallOfFame(1)
hof3 = tools.HallOfFame(1)
hof4 = tools.HallOfFame(1)

# Simply call the Evolve() function as follows
pop, log = Evolve([pop1, pop2, pop3, pop4], toolbox, 0.5, 0.4,generations, halloffame=[hof1, hof2, hof3, hof4], verbose=True)

# Note that pop is now a list, containing the 4 populations i.e., pop = [pop1, pop2, pop3, pop4]. Same is the case with log.
