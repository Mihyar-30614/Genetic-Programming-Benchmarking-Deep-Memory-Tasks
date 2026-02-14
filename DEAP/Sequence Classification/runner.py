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

def get_args():
    str = "\n************************************************************\n"
    str += "*           Welcome to Copy Task champion arena             *\n"
    str += "*  Please provide the following arguments comma delimited   *\n"
    str += "*  Type of test to run options are (required):              *\n"
    str += "*       - std -> to run the standard champion               *\n"
    str += "*       - mul -> to run the multiplication champion         *\n"
    str += "*       - mod -> to run the modified champion               *\n"
    str += "*       - log -> to run the logical champion                *\n"
    str += "*   Depth of sequence i.e. number of 1/-1's (required):     *\n"
    str += "*       - options are: 4, 5, 6, 15, 21                      *\n"
    str += "*   Range of noise to use (required):                       *\n"
    str += "*       - options are: 0, 0.5, 0.25, 0.125                  *\n"
    str += "*   Which champion to load (optional):                      *\n"
    str += "*       - example 'champion_1' .... 'champion_50'           *\n"
    str += "*   Number of tests to run (optional):                      *\n"
    str += "*       - integer represents the number of tests            *\n"
    str += "*   Length of Noise in sequence (optional):                 *\n"
    str += "*       - integer represents the length of noise            *\n"
    str += "************************************************************\n"
    print(str)
    
    options = ("std", "mul","mod","log")
    while True:
        try:
            input_args = input("Choose your champion:\n").strip().lower().split(",")

            if len(input_args) < 2:
                raise ValueError

            if len(input_args)>0 and input_args[0].strip() not in options:
                raise ValueError

            if len(input_args)>1 and int(input_args[1].strip()) not in (4, 5, 6, 15, 21):
                raise ValueError

            # Everything is fine 
            break

        except ValueError:
            print("Sorry your entry is wrong, try again!")

    # Reading Type and Depth Values
    type = input_args[0]
    depth = int(input_args[1])

    # Default Range Value if not passed
    range_val = 0
    if type in ('mod'):
        if len(input_args) > 2:
            range_val = float(input_args[2])
        else:
            range_val = 0.5

    # Default Champion if not passed
    champion = "champion_1"
    if len(input_args) > 3:
        champion = input_args[3]
    
    # Default Number of tests if not passed
    num_test = 50
    if len(input_args) > 4:
        num_test = int(input_args[4])

    # Default Noise and generalize
    noise, generalize = 10, True
    if len(input_args) > 5:
        noise = int(input_args[5])
        generalize = False

    return type, depth, range_val, champion, num_test, generalize, noise

'''
Problem setup
'''

# Generate Random Data
def generate_data(noise, depth, range_val, num_tests, generalize):
    retval = []
    for _ in range(num_tests):
        sequence = []
        sequence.append(random.choice((-1.0, 1.0)))
        noise = 10 if not generalize else random.randint(10, 20)
        for _ in range(depth - 1):
            sequence.extend([random.uniform(-range_val,range_val) for _ in range(noise)])
            sequence.append(random.choice((-1.0, 1.0)))
        retval.append(sequence)
    return retval

# Generate Classification based on dataset
def generate_output(dataset, type):
    retval = []
    for i in range(num_tests):
        data = dataset[i]
        sequence = []
        counter = 0
        for el in data:
            if type == 'mod':
                if el == 1 or el == -1:
                    counter += el
            else:
                counter += el
            sequence.append(-1 if counter < 0 else 1)
        retval.append(sequence)
    return retval

# Generate expected GP Action based on Dataset
def generate_action(dataset, type):
    retval = []
    for i in range(num_tests):
        data = dataset[i]
        sequence = []
        MEMORY = []
        if type == 'mod':
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
        else:
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

# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2

def create_gp(type):
    # defined a new primitive set for strongly typed GP
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 2), float)

    if type in ("std", "vec", "mod"):
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(protected_div, [float, float], float)

    if type == "mul":
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)

    if type == "log":
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
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("compile", gp.compile, pset=pset)
    return toolbox

if __name__ == "__main__":

    # Const variables
    local_dir = os.path.dirname(__file__)
    champ_path = os.path.join(local_dir, 'champions/')

    # Get input from terminal
    type, depth, range_val, champion, num_tests, generalize, noise = get_args()

    # Generate Data
    data_validation = generate_data(noise, depth, range_val, num_tests, generalize)
    labels_validation = generate_output(data_validation, type)
    actions_validation = generate_action(data_validation, type)
    
    # Create GP
    toolbox = create_gp(type)
    
    # Load Champion
    champ_name = champ_path + str(depth) + '_champions_' + type
    with open(champ_name, 'rb') as f:
        champions = pickle.load(f)
        print("loaded champions")

    hof1, hof2, hof3, hof4 = champions[champion]

    
    # Running Test on unseen data and checking results
    print("\n==================")
    print("Begin Testing ....")
    print("==================\n")

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
        accuracy = accuracy_score(labels_validation[i], predictions[i])
        print("Test #{} Accuracy: {}".format(i, accuracy))
        total_accuracy += accuracy
    
    print("------------------------")
    print("Total Accuracy: {}".format(total_accuracy/num_tests))