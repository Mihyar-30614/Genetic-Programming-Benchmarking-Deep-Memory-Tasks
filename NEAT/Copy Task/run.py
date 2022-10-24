from __future__ import division, print_function

import os
import neat
import numpy as np
import pickle
import random
from sklearn.metrics import accuracy_score

# Data Config
seq_length = 10         # length of the test sequence.
bits = 8                # number of bits used
num_tests = 50          # num_tests is the number of random examples each network is tested against.
num_runs = 50           # number of runs

# Results Config
generalize = False
save_log = False
detail_log = False
results = []
overall = []

# Directory of files
local_dir = os.path.dirname(__file__)
rpt_path = os.path.join(local_dir, 'reports/')
champ_path = os.path.join(local_dir, 'champions/')
config_path = os.path.join(local_dir, 'config')

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

if __name__ == "__main__":

    # Determine path to configuration file.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # Load Champion
    print("Loading champions ...")
    champ_name = champ_path + str(bits) + '_champions_std'
    with open(champ_name, 'rb') as f:
        champions = pickle.load(f)
        print("loaded champions")

    for ch in range(num_runs):

        print("Loading Champion {} ....".format(ch+1))
        champion_name = "champion_" + str(ch+1)
        champion = champions[champion_name]

        print("Generate Test Dataset ...")
        data_validation = generate_data(seq_length, num_tests, bits, generalize)
        actions_validation = generate_action(data_validation, num_tests)
        
        '''
        Running Test on unseen data and checking results
        '''
        print("Begin Testing ....")

        winner_net = neat.nn.RecurrentNetwork.create(champion, config)
        # Evaluate the sum of correctly identified
        predict_actions = []
        # Evaluate the sum of correctly identified
        for i in range(num_tests):
            data, actions = data_validation[i], []
            length = len(data)
            prog_state = 0
            winner_net.reset()

            for j in range(length):
                net_input = [data[j][0], data[j][1], prog_state]
                outdata = winner_net.activate(net_input)
                arg1 = outdata[0]
                arg2 = outdata[1]
                arg3 = outdata[2]
                arg4 = outdata[3]
                prog_state = outdata[4]
                pos = np.argmax([arg1, arg2, arg3, arg4])
                actions.append(pos)

            predict_actions.append(actions)

        # Evaluate predictions
        total_accuracy = 0
        for i in range(num_tests):
            accuracy = accuracy_score(actions_validation[i], predict_actions[i])
            results.append(accuracy)
            total_accuracy += accuracy
            if detail_log:
                print("Delim1: \n{}".format([item[0] for item in data_validation[i]]))
                print("Delim2: \n{}".format([item[1] for item in data_validation[i]]))
                print("Prediction Actions: \n{}".format(predict_actions[i]))
                print("Actions: \n{}".format(actions_validation[i]))
                print("Accuracy: {}".format(accuracy))
            print("Champion {} Test {} Accuracy: {}".format(ch+1, i+1, accuracy))
        print("==================================================================")
        overall.append(total_accuracy)

    print("Overall Champions Stats")
    for idx, res in enumerate(overall):
        print("Champion {} Accuracy: {}".format(idx+1, res))
    print("==================================================================")

    '''
    Calculate Mean and Standard Deviation
    '''
    results_mean = np.mean(results, axis=0)
    results_std = np.std(results, axis=0)

    '''
    View Last item as result
    '''
    print("Copy Task Champions overall performance")
    print("Mean: {:.2f}%".format(results_mean * 100))
    print("STD: {:.2f}".format(results_std * 100))

    if save_log:
        with open(rpt_path + 'run_results_' + str(seq_length), 'wb') as f:
            pickle.dump(results, f)