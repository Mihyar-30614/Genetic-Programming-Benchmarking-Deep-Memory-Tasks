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
    str += "*       - vec -> to run the full vector champion            *\n"
    str += "*   Which champion to load (optional):                      *\n"
    str += "*       - example 'champion_1' .... 'champion_50'           *\n"
    str += "*   Number of tests to run (optional):                      *\n"
    str += "*       - integer represents the number of tests            *\n"
    str += "*   Sequence length to remember (optional):                 *\n"
    str += "*       - integer represents the length of sequence         *\n"
    str += "************************************************************\n"
    print(str)
    
    options = ("std", "mul","mod","log","vec")
    while True:
        try:
            input_args = input("Choose your champion:\n").strip().lower().split(",")
            if input_args[0].strip() not in options:
                raise ValueError
            else:
                break
        except ValueError:
            print("Sorry your entry is wrong, try again!")

    # Reading Type and Depth Values
    type = input_args[0]

    # Default Champion if not passed
    champion = "champion_1"
    if len(input_args) > 1:
        champion = input_args[1]
    
    # Default Number of tests if not passed
    num_test = 50
    if len(input_args) > 2:
        num_test = int(input_args[2])

    # Default seq_length and generalize
    seq_length, generalize = 10, True
    if len(input_args) > 3:
        seq_length = int(input_args[3])
        generalize = False

    return type, champion, num_test, generalize, seq_length

'''
Problem setup
'''

def generate_data(seq_length, num_tests, bits, generalize, type):
    retval = []
    for _ in range(num_tests):
        if generalize:
                seq_length = random.randint(10, 20)
        # Adding 2 to bits for writing delim and reading delim
        # also adding 2 to length for delim sequence
        sequence = np.zeros([seq_length + 2, bits + 2], dtype=np.float32)
        for idx in range(1, seq_length + 1):
            sequence[idx, 2:bits+2] = np.random.rand(bits).round()

        sequence[0, 0] = 1                # Setting Wrting delim
        sequence[seq_length+1, 1] = 1     # Setting reading delim

        if type == "mod":
            sequence[0, 1] = -1               # Setting Wrting delim
            sequence[seq_length+1, 0] = -1    # Setting reading delim

        recall = np.zeros([seq_length, bits + 2], dtype=np.float32)
        data = np.concatenate((sequence, recall), axis=0).tolist()
        retval.append(data)
    return retval

def generate_action(data_array, num_tests, type):
    retval = []
    delim = -1 if type == 'mod' else 0

    for i in range(num_tests):
        data, action, write = data_array[i], [], False
        length = len(data)

        # 0 = PUSH, 1 = POP HEAD, 2 = NOTHING, 3 = POP TAIL
        for x in range(length):
            if data[x][0] == 1 and data[x][1] == delim:
                write = True
                action.append(2)
            elif data[x][0] == delim and data[x][1] == 1:
                write = False
                action.append(2)
            else:
                action.append(0) if write == True else action.append(1)
        retval.append(action)
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
    if type in ("vec"):
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, bits + 3), float)
    else:
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 3), float)

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
    bits = 8

    # Get input from terminal
    type, champion, num_tests, generalize, seq_length = get_args()

    # Generate Data
    data_validation = generate_data(seq_length, num_tests, bits, generalize, type)
    actions_validation = generate_action(data_validation, num_tests, type)
    
    # Create GP
    toolbox = create_gp(type)
    
    # Load Champion
    champ_name = champ_path + str(bits) + '_champions_' + type
    with open(champ_name, 'rb') as f:
        champions = pickle.load(f)
        print("loaded champions")

    hof1, hof2, hof3, hof4, hof5 = champions[champion]

    
    # Running Test on unseen data and checking results
    print("\n==================")
    print("Begin Testing ....")
    print("==================\n")

    # Transform the tree expression in a callable function
    tree1 = toolbox.compile(expr=hof1)
    tree2 = toolbox.compile(expr=hof2)
    tree3 = toolbox.compile(expr=hof3)
    tree4 = toolbox.compile(expr=hof4)
    tree5 = toolbox.compile(expr=hof5)

    # Evaluate the sum of correctly identified
    predict_actions = []
    # Evaluate the sum of correctly identified
    for i in range(num_tests):
        data, actions = data_validation[i], []
        length = len(data)
        prog_state = 0

        for j in range(length):
            if type in ("vec"):
                arg1 = tree1(*data[j], prog_state)
                arg2 = tree2(*data[j], prog_state)
                arg3 = tree3(*data[j], prog_state)
                arg4 = tree4(*data[j], prog_state)
                prog_state = tree5(*data[j], prog_state)
            else:
                arg1 = tree1(data[j][0], data[j][1], prog_state)
                arg2 = tree2(data[j][0], data[j][1], prog_state)
                arg3 = tree3(data[j][0], data[j][1], prog_state)
                arg4 = tree4(data[j][0], data[j][1], prog_state)
                prog_state = tree5(data[j][0], data[j][1], prog_state)
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