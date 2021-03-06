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
seq_lengths = [50,100]  # length of the test sequence.
bits = 8                # number of bits used
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

def generate_data(seq_length):
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

def generate_action(data_array):
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

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, bits + 3), float)

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
    champ_name = champ_path + str(bits) + '_champions_vec'
    with open(champ_name, 'rb') as f:
        champions = pickle.load(f)
        print("loaded champions")

    for seq_length in seq_lengths:

        print("Generalizing {} Sequence Length".format(seq_length))

        for ch in range(num_runs):

            print("Loading Champion {} ....".format(ch+1))
            champion = "champion_" + str(ch+1)
            hof1, hof2, hof3, hof4, hof5 = champions[champion]

            print("Generate Test Dataset ...")
            data_validation = generate_data(seq_length)
            actions_validation = generate_action(data_validation)
            
            print("Begin Testing ....")

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
                    arg1 = tree1(*data[j], prog_state)
                    arg2 = tree2(*data[j], prog_state)
                    arg3 = tree3(*data[j], prog_state)
                    arg4 = tree4(*data[j], prog_state)
                    prog_state = tree5(*data[j], prog_state)
                    pos = np.argmax([arg1, arg2, arg3, arg4])
                    actions.append(pos)

                predict_actions.append(actions)

            # Evaluate predictions
            for i in range(num_tests):
                accuracy = accuracy_score(actions_validation[i], predict_actions[i])
                results.append(accuracy)
                print("Champion {} Test {} Accuracy: {}".format(ch+1, i+1, accuracy))
            print("==================================================================")
        
        if save_log:
            with open(rpt_path + 'gen_vec_results_' + str(seq_length), 'wb') as f:
                pickle.dump(results, f)
        