from __future__ import division, print_function

import os
import neat
import numpy as np
import pickle
import random
from sklearn.metrics import accuracy_score

# Data Config
depth = 21              # Number of (1, -1) in a sequence
corridor_length = 10    # Number of Zeros between values
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

if __name__ == "__main__":

    # Determine path to configuration file.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # Load Champion
    print("Loading champions ...")
    champ_name = champ_path + str(depth) + '_champions_std'
    with open(champ_name, 'rb') as f:
        champions = pickle.load(f)
        print("loaded champions")

    for ch in range(num_runs):

        print("Loading Champion {} ....".format(ch+1))
        champion_name = "champion_" + str(ch+1)
        champion = champions[champion_name]

        print("Generate Test Dataset ...")
        data_validation = generate_data(depth, corridor_length)
        actions_validation = generate_action(data_validation)
        
        '''
        Running Test on unseen data and checking results
        '''
        print("Begin Testing ....")

        winner_net = neat.nn.RecurrentNetwork.create(champion, config)
        predict_actions = []
        # Evaluate the sum of correctly identified
        for i in range(num_tests):
            instructions, data, actions = data_validation[i][0], data_validation[i][1], []
            length = len(data)

            for j in range(length):
                net_input = [instructions[j], data[j]]
                outdata = winner_net.activate(net_input)
                arg1 = outdata[0]
                arg2 = outdata[1]
                arg3 = outdata[2]
                arg4 = outdata[3]
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
                print("instructrions: \n{}".format(data_validation[i][0]))
                print("data: \n{}".format(data_validation[i][1]))
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
        with open(rpt_path + 'run_results_' + str(depth), 'wb') as f:
            pickle.dump(results, f)