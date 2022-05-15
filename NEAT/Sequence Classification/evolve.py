"""
This is an example of sequence classification using NEAT.

Example Input:
    sequence        = [1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0]
    Stack_output    = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0]
    
Example Output:
    Action_output   = [0.0, 2.0, 2.0, 0.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 0.0, 2.0] where 0=PUSH, 1=POP, 2=NONE
    Stack_output    = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0] where 0 means empty
    classification  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
"""

import multiprocessing
import os
import visualize
import neat
import numpy as np
import random
import pickle

# Data Config
depth = 4              # Number of (1, -1) in a sequence
noise = 10              # Number of Zeros between values
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

# Generate Random Data
def generate_data(depth, noise):
    retval = []
    for _ in range(num_tests):
        sequence = []
        sequence.append(random.choice((-1.0, 1.0)))
        for _ in range(depth - 1):
            sequence.extend([0 for _ in range(noise)])
            sequence.append(random.choice((-1.0, 1.0)))
        retval.append(sequence)
    return retval

# Generate Classification based on dataset
def generate_output(dataset):
    retval = []
    for i in range(num_tests):
        data = dataset[i]
        sequence = []
        counter = 0
        for el in data:
            counter += el
            sequence.append(-1 if counter < 0 else 1)
        retval.append(sequence)
    return retval

# Generate expected GP Action based on Dataset
def generate_action(dataset):
    retval = []
    for i in range(num_tests):
        data = dataset[i]
        sequence = []
        MEMORY = []
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

# Generate Train Dataset
random_noise = noise

if generalize:
    random_noise = random.randint(10, 20)
data_train = generate_data(depth, random_noise)
labels_train = generate_output(data_train)
actions_train = generate_action(data_train)


'''
    Begining of NEAT Structure
'''

def eval_function(genome, config):
    
    net = neat.nn.RecurrentNetwork.create(genome, config)

    fitness, total_len = 0, 0
    # Evaluate the sum of correctly identified
    for i in range(num_tests):
        data, labels, actions = data_train[i], labels_train[i], actions_train[i]
        MEMORY, classification = [], []
        counter, stopped = 0, False
        length = len(data)
        total_len += length
        net.reset()

        for j in range(length):
            # If stack is empty then 0, else the value on top of stack
            stack_output = MEMORY[counter - 1] if counter > 0 else 0

            net_input = [data[j],stack_output]
            outdata = net.activate(net_input)
            arg1 = outdata[0]
            arg2 = outdata[1]
            arg3 = outdata[2]
            pos = np.argmax([arg1, arg2, arg3])

            if pos == actions[j]:
                # correct action produced
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
                if temp == labels[j]:
                    classification.append(temp)
                else:
                    print("Something has went horribly wrong!")
            else:
                # wrong action produced
                fitness += len(classification)
                stopped = True
                break
        if stopped == False:
            fitness += len(classification)

    return (fitness/total_len) * 100


if __name__ == "__main__":

    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
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
    with open(champ_path + str(depth) + '_champions_std', 'wb') as f:
        pickle.dump(champions, f)

    if save_log:
        with open(rpt_path + str(depth) + '_report_std', 'wb') as f:
            pickle.dump(reports, f)