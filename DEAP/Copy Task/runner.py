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
    str += "*       - gen -> to run the generalization of task          *\n"
    str += "*   Which champion to load (optional):                      *\n"
    str += "*       - example 'champion_1' .... 'champion_50'           *\n"
    str += "*   Number of tests to run (optional):                      *\n"
    str += "*       - integer represents the number of tests            *\n"
    str += "************************************************************\n"
    print(str)
    
    while True:
        try:
            input_args = input("Choose your champion:\n").strip().lower().split(",")
            if input_args[0].strip() not in ("std", "mul","mod","log","gen"):
                raise ValueError
            else:
                break
        except ValueError:
            print("Sorry your entry is wrong, try again!")

    return input_args

'''
Problem setup
'''

def generate_data(seq_length, num_tests, bits, generalize):
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

        recall = np.zeros([seq_length, bits + 2], dtype=np.float32)
        data = np.concatenate((sequence, recall), axis=0).tolist()
        retval.append(data)
    return retval

def generate_action(data_array, num_tests):
    retval = []
    for i in range(num_tests):
        data, action, write, read = data_array[i], [], False, False
        length = len(data)

        # 0 = PUSH, 1 = POP HEAD, 2 = NOTHING, 3 = POP TAIL
        for x in range(length):
            if data[x][0] == 1 and data[x][1] == 0:
                write = True
                read = False
                action.append(2)
            elif data[x][0] == 0 and data[x][1] == 1:
                write = False
                read = True
                action.append(2)
            else:
                if write == True:
                    action.append(0)
                elif read == True:
                    action.append(1)
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

def create_gp(type):
    print("Creating GP ...")
    # defined a new primitive set for strongly typed GP
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 3), float)

    # Float operators
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    
    if type != "mul":
        pset.addPrimitive(protected_div, [float, float], float)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("compile", gp.compile, pset=pset)
    print("GP Created!")
    return toolbox

if __name__ == "__main__":
    args = get_args()
    print(args)