"""
This is an example of Copy Task using NEAT-Python (1-bit).

Example Input:
    for sequence length of 3, Write delim is in first position, Read delim is in second position.
    Sample Input = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
Example Output:
    Abstracted output, if NEAT guess the command right the stack will be correct. This is why we only care about commands.
    Sample Output = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
"""

import multiprocessing
import os
import visualize
import neat
import numpy as np
import random
import pickle

# length of the test sequence.
seq_length = 10
# number of bits used
bits = 1
# num_tests is the number of random examples each network is tested against.
num_tests = 50
num_generations = 500
generalize = True
save_log = False

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
            prog_state = outdata[3]
            pos = np.argmax([arg1, arg2, arg3])

            if pos == actions[j]:
                fitness += 1
            else:
                # wrong action produced
                break
    return fitness/total_len


if __name__ == "__main__":
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'CopyTask_config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_function)
    winner = pop.run(pe.evaluate, num_generations)

    # Log statistics.
    stats.save()

    # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the winner
    with open('champion-gnome', 'wb') as f:
        pickle.dump(winner, f)