"""
This is an example of sequence recall using DEAP.

Example Input:
    sequence        = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Stack_output    = [1.0, -1.0, -1.0, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    
Example Output:
    Action_output   = [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1] where 0=PUSH, 1=POP HEAD, 2=NONE, 4=POP TAIL
"""

import multiprocessing
import os
import visualize
import neat
import numpy as np
import random
import pickle
import shutil

# Number of (1, -1) in a sequence
depth = 21
# Number of Zeros between values
corridor_length = 10
# num_tests is the number of random examples each network is tested against.
num_tests = 50
num_generations = 500
generalize = True
save_log = False

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

data_train = generate_data(depth, corridor_length)
actions_train = generate_action(data_train)

'''
    Begining of NEAT Structure
'''

def eval_function(genome, config):
    
    net = neat.nn.RecurrentNetwork.create(genome, config)

    fitness, total_len = 0, 0
    # Evaluate the sum of correctly identified
    for i in range(num_tests):
        instructions, data, actions = data_train[i][0], data_train[i][1], actions_train[i]
        length = len(data)
        total_len += length
        net.reset()

        for j in range(length):
            net_input = [instructions[j], data[j]]
            outdata = net.activate(net_input)
            arg1 = outdata[0]
            arg2 = outdata[1]
            arg3 = outdata[2]
            arg4 = outdata[3]
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
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # for i in range(1,21):
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

    # Add to reporting
    if save_log:
        path = os.path.join(local_dir, str(depth)+'-deep-report/')
        src_dir = local_dir + 'fitness_history.csv'
        dest_dir = path + str(depth) + '-progress_report' + str(i) + ".csv"
        shutil.copy(src_dir,dest_dir)