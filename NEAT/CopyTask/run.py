from __future__ import division, print_function

import os
import neat
import numpy as np
import pickle
import random
from sklearn.metrics import accuracy_score

# length of the test sequence.
seq_length = 10
# number of bits used
bits = 8
# num_tests is the number of random examples each network is tested against.
num_tests = 50
generalize = True


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

data_validation = generate_data(seq_length)
actions_validation = generate_action(data_validation)


def run(config, winner):

    '''
    Running Test on unseen data and checking results
    '''

    print("\n==================")
    print("Begin Testing ....")
    print("==================\n")

    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

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
        print("Delim1: \n{}".format([item[0] for item in data_validation[i]]))
        print("Delim2: \n{}".format([item[1] for item in data_validation[i]]))
        print("Prediction Actions: \n{}".format(predict_actions[i]))
        print("Actions: \n{}".format(actions_validation[i]))
        accuracy = accuracy_score(actions_validation[i], predict_actions[i])
        print("Accuracy: {}".format(accuracy))
        print("==================================================================")
        total_accuracy += accuracy
    
    print("Total Accuracy: {}".format(total_accuracy/num_tests))


if __name__ == "__main__":

    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'CopyTask_config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # Load the winner
    with open('champion-gnome', 'rb') as f:
        winner = pickle.load(f)
        print("loaded Genome:")
        print(winner)

    # Run Test
    run(config, winner)
