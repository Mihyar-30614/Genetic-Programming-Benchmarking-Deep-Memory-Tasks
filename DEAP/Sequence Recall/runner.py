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
    str += "*   Which champion to load (optional):                      *\n"
    str += "*       - example 'champion_1' .... 'champion_50'           *\n"
    str += "*   Number of tests to run (optional):                      *\n"
    str += "*       - integer represents the number of tests            *\n"
    str += "*   Length of Corridor in sequence (optional):              *\n"
    str += "*       - integer represents the length of noise            *\n"
    str += "************************************************************\n"
    print(str)
    
    options = ("std", "mul","mod","log")
    while True:
        try:
            valid = True
            input_args = input("Choose your champion:\n").strip().lower().split(",")

            if input_args[0].strip() not in options:
                valid = False

            if int(input_args[1].strip()) not in (4, 5, 6, 15, 21):
                valid= False

            if valid:
                break
            else:
                raise ValueError
        except ValueError:
            print("Sorry your entry is wrong, try again!")

    # Setting values and default values
    type = input_args[0]
    depth = int(input_args[1])
    champion = input_args[2] if len(input_args) >= 3 else "champion_1"
    num_test = int(input_args[3]) if len(input_args) >= 4 else 50
    noise = int(input_args[4]) if len(input_args) >= 5 else 10
    generalize = False if len(input_args) >= 5 else True

    return type, depth, champion, num_test, generalize, noise

'''
Problem setup
'''

# Generate Random Data
# def generate_data(noise, depth, range_val, num_tests, generalize):
def generate_data(depth, corridor_length, num_tests, generalize):
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

# Generate expected GP Action based on Dataset
def generate_action(data_array, type, num_tests):
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

# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2

def create_gp(type):
    # defined a new primitive set for strongly typed GP
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 2), float)

    if type in ("std", "mod"):
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
    type, depth, champion, num_tests, generalize, corridor_length = get_args()

    # Generate Data
    data_validation = generate_data(depth, corridor_length, num_tests, generalize)
    actions_validation = generate_action(data_validation, type, num_tests)
    
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
    total_accuracy = 0
    for i in range(num_tests):
        accuracy = accuracy_score(actions_validation[i], predict_actions[i])
        print("Test #{} Accuracy: {}".format(i, accuracy))
        total_accuracy += accuracy
    
    print("------------------------")
    print("Total Accuracy: {}".format(total_accuracy/num_tests))