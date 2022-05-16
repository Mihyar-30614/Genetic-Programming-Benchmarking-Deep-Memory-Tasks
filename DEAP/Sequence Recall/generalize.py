import itertools
import operator
import random
import numpy as np
import pickle
import os

from sklearn.metrics import accuracy_score
from deap import gp
from deap import base
from deap import creator

depth = 100
corridor_length = 10
num_tests = 50
generalize = True
num_runs = 20
local_dir = os.path.dirname(__file__)
champ_path = os.path.join(local_dir, 'champion/')
results = []

'''
Problem setup
'''

def generate_data(depth, corridor_length):
    retval = []
    for _ in range(num_tests):
        data1, data2 = [], []
        # create insturctions
        for _ in range(depth):
            data1.append(1)
            data2.append(random.choice((-1.0, 1.0)))

        # create maze
        for _ in range(depth):
            if generalize:
                corridor_length = random.randint(10, 20)

            countdown = 1
            step = round(countdown/corridor_length, 2)

            while countdown >= 0:
                # Countdown starts with 1 and decrease
                countdown = round(countdown, 2)
                data1.append(0)
                data2.append(countdown)
                countdown -= step
            # Just in case Countdown didn't reach 0
            if data2[-1] != 0:
                data1.append(0)
                data2.append(0)

        retval.append([data1, data2])
    return retval

def generate_action(data_array):
    retval = []
    for i in range(num_tests):
        output, instruction, data = [], data_array[i][0], data_array[i][1]
        for x in range(len(instruction)):
            # 0 = PUSH, 1 = POP HEAD, 2 = NOTHING, 3 = POP TAIL
            if instruction[x] == 1:
                output.append(0)
            elif instruction[x] == 0 and data[x] == 0:
                output.append(1)
            else:
                output.append(2)
        retval.append(output)
    return retval


'''
    Begining of DEAP Structure
'''

# Define a protected division function
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

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

if __name__ == "__main__":

    for ch in range(num_runs):

        print("Loading Champion {} ....".format(ch+1))

        with open(champ_path + 'output1_' + str(ch+1), 'rb') as f:
            hof1 = pickle.load(f)

        with open(champ_path + 'output2_' + str(ch+1), 'rb') as f:
            hof2 = pickle.load(f)

        with open(champ_path + 'output3_' + str(ch+1), 'rb') as f:
            hof3 = pickle.load(f)

        with open(champ_path + 'output4_' + str(ch+1), 'rb') as f:
            hof4 = pickle.load(f)

        print("Generate Test Dataset ...")
        data_validation = generate_data(depth, corridor_length)
        actions_validation = generate_action(data_validation)
        
        print("Begin Testing ....")

        # Transform the tree expression in a callable function
        tree1 = toolbox.compile(expr=hof1)
        tree2 = toolbox.compile(expr=hof2)
        tree3 = toolbox.compile(expr=hof3)
        tree4 = toolbox.compile(expr=hof4)

        # Evaluate the sum of correctly identified
        predict_actions = []
        # Evaluate the sum of correctly identified
        for i in range(num_tests):
            instructions, data, actions = data_validation[i][0], data_validation[i][1], []
            length = len(data)

            for j in range(length):
                arg1 = tree1(instructions[j], data[j])
                arg2 = tree2(instructions[j], data[j])
                arg3 = tree3(instructions[j], data[j])
                arg4 = tree4(instructions[j], data[j])
                pos = np.argmax([arg1, arg2, arg3, arg4])
                actions.append(pos)

            predict_actions.append(actions)

        # Evaluate predictions
        for i in range(num_tests):
            accuracy = accuracy_score(actions_validation[i], predict_actions[i])
            results.append(accuracy)
            print("Champion {} Test {} Accuracy: {}".format(ch+1, i+1, accuracy))
        print("==================================================================")
    
    # Save the results
    with open(champ_path + 'results_' + str(depth), 'wb') as f:
        pickle.dump(results, f)
        