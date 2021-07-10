import itertools
import operator
import random
import numpy as np
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from deap import gp
from deap import base
from deap import creator

# Loading the Test Dataset
lines = np.loadtxt("Test-dataset.txt", comments="#", delimiter=",", unpack=False, dtype=float)
data_validation = lines[:,0:9]
labels_validation = lines[:,9:].flatten()

'''
    Begin DEAP Structure
'''

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

# Load the best tree
with open('10k-output1', 'rb') as f:
    hof1 = pickle.load(f)
    print("loaded Tree1:")
    print(hof1)

with open('10k-output2', 'rb') as f:
    hof2 = pickle.load(f)
    print("loaded Tree2:")
    print(hof2)

if __name__ == "__main__":
    '''
    Running Test on unseen data and checking results
    '''

    print("\n==================")
    print("Begin Testing ....")
    print("==================\n")
    # Transform the tree expression in a callable function
    tree1 = toolbox.compile(expr=hof1)
    tree2 = toolbox.compile(expr=hof2)

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