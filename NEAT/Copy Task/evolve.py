"""
This is an example of Copy Task using NEAT-Python (1-bit).

Example Input:
    for sequence length of 3, Write delim is in first position, Read delim is in second position.
    Sample Input = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
Example Output:
    Abstracted output, if NEAT guess the command right the stack will be correct. This is why we only care about commands.
    Sample Output = [[2, 0, 0, 0, 2, 1, 1, 1]]
"""

import multiprocessing
import os
import visualize
import neat
import numpy as np
import random
import pickle

# Data Config
seq_length = 10         # length of the test sequence.
bits = 8                # number of bits used
num_tests = 50          # num_tests is the number of random examples each network is tested against.
num_runs = 50           # number of runs

# Results Config
generalize = True
save_log = True
verbose_val = False
num_generations = 500

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
        

data_train = generate_data(seq_length)
actions_train = generate_action(data_train)


'''
    Begining of NEAT Structure
'''

def eval_function(genome, config):
    
    net = neat.nn.RecurrentNetwork.create(genome, config)
    
    fitness, total_len = 0, 0
    # Evaluate the sum of correctly identified
    for i in range(num_tests):
        data, actions = data_train[i], actions_train[i]
        length = len(data)
        total_len += length
        prog_state = 0
        net.reset()
        
        for j in range(length):
            net_input = [data[j][0], data[j][1], prog_state]
            outdata = net.activate(net_input)
            arg1 = outdata[0]
            arg2 = outdata[1]
            arg3 = outdata[2]
            arg4 = outdata[3]
            prog_state = outdata[4]
            pos = np.argmax([arg1, arg2, arg3, arg4])

            if pos == actions[j]:
                fitness += 1
            else:
                # wrong action produced
                break
    return (fitness/total_len) * 100


if __name__ == "__main__":

    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'CopyTask_config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    champions, reports = {}, {}
    for i in range(num_runs):
        pop = neat.Population(config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(verbose_val))

        if not verbose_val:
            print("Run #: " + str(i+1))
        
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_function)
        winner = pop.run(pe.evaluate, num_generations)

        # Save the winner
        champions["champion_" + str(i+1)] = winner
        progress_report = [c.fitness for c in stats.most_fit_genomes]
        reports['report' + str(i+1)] = progress_report

    # Save Champions
    with open(champ_path + str(bits) + '_champions_std', 'wb') as f:
        pickle.dump(champions, f)

    if save_log:
        with open(rpt_path + str(bits) + '_report_std', 'wb') as f:
            pickle.dump(reports, f)