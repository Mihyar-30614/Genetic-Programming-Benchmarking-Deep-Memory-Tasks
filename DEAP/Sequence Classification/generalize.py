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

# Data Config
depths = [50,100]       # Number of (1, -1) in a sequence
noise = 10              # Number of Zeros between values
num_tests = 50          # num_tests is the number of random examples each network is tested against.
num_runs = 50           # number of runs

# Results Config
generalize = False
save_log = True
results = []

# Directory of files
local_dir = os.path.dirname(__file__)
rpt_path = os.path.join(local_dir, 'reports/')
champ_path = os.path.join(local_dir, 'champions/')

'''
Problem setup
'''

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

    # Load Champion
    print("Loading champions ...")
    champ_name = champ_path + '21_champions_std'
    with open(champ_name, 'rb') as f:
        champions = pickle.load(f)
        print("loaded champions")

    for depth in depths:

        print("Generalizing {} Sequence Length".format(depth))

        for ch in range(num_runs):

            print("Loading Champion {} ....".format(ch+1))
            champion = "champion_" + str(ch+1)
            hof1, hof2, hof3, hof4 = champions[champion]

            print("Generate Test Dataset ...")
            # Generate Test Dataset
            random_noise = noise

            if generalize:
                random_noise = random.randint(10, 20)
            data_validation = generate_data(depth, random_noise)
            labels_validation = generate_output(data_validation)
            actions_validation = generate_action(data_validation)
            
            print("Begin Testing ....")

            # Transform the tree expression in a callable function
            tree1 = toolbox.compile(expr=hof1)
            tree2 = toolbox.compile(expr=hof2)
            tree3 = toolbox.compile(expr=hof3)
            tree4 = toolbox.compile(expr=hof4)

            # Evaluate the sum of correctly identified
            predictions, predict_actions = [],[]
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
                    temp = 1 if stack_output >= 0 else -1
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

            # Evaluate predictions
            for i in range(num_tests):
                accuracy = accuracy_score(labels_validation[i], predictions[i])
                results.append(accuracy)
                print("Champion {} Test {} Accuracy: {}".format(ch+1, i+1, accuracy))
            print("==================================================================")
        
        # Save the results
        if save_log:
            with open(rpt_path + 'gen_results_' + str(depth), 'wb') as f:
                pickle.dump(results, f)