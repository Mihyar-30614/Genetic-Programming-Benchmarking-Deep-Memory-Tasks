import itertools
import operator
import random
import numpy as np
import pickle

from sklearn.metrics import accuracy_score
from deap import gp
from deap import base
from deap import creator

# Number of (1, -1) in a sequence
depth = 21
# Number of Zeros between values
noise = 10
# num_tests is the number of random examples each network is tested against.
num_tests = 50
gneralize = True

# Generate Random Data
def generate_data(depth, noise):
    retval = []
    for _ in range(num_tests):
        sequence = []
        sequence.append(random.choice((-1.0, 1.0)))
        for _ in range(depth - 1):
            sequence.extend([random.uniform(-0.50,0.50) for _ in range(noise)])
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
            if el == 1 or el == -1:
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
            if el != 1 and el != -1:
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

# Generate Test Dataset
random_noise = noise

if gneralize:
    random_noise = random.randint(10, 20)
data_validation = generate_data(depth, random_noise)
labels_validation = generate_output(data_validation)
actions_validation = generate_action(data_validation)

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


# Load the best tree
with open('output1', 'rb') as f:
    hof1 = pickle.load(f)
    print("loaded Tree1:")
    print(hof1)

with open('output2', 'rb') as f:
    hof2 = pickle.load(f)
    print("loaded Tree2:")
    print(hof2)

with open('output3', 'rb') as f:
    hof3 = pickle.load(f)
    print("loaded Tree3:")
    print(hof3)


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
    tree3 = toolbox.compile(expr=hof3)

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
            pos = np.argmax([arg1, arg2, arg3])

            # Action has been decided
            temp = 1 if stack_output >= 0 else -1
            actions.append(pos)
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
            classification.append(temp)

        predictions.append(classification)
        predict_actions.append(actions)

    # Evaluate predictions
    total_accuracy = 0
    for i in range(num_tests):
        print("Predictions: \n{}".format(predictions[i]))
        print("labels: \n{}".format(labels_validation[i]))
        print("Prediction Actions: \n{}".format(predict_actions[i]))
        print("Actions: \n{}".format(actions_validation[i]))
        accuracy = accuracy_score(labels_validation[i], predictions[i])
        print("Accuracy: {}".format(accuracy))
        print("==================================================================")
        total_accuracy += accuracy
    
    print("Total Accuracy: {}".format(total_accuracy/num_tests))
