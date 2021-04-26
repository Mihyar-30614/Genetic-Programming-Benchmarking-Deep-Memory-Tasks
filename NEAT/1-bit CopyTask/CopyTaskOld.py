from __future__ import division, print_function
from random import randint, random

import multiprocessing
import os
import visualize
import neat
import numpy as np

# length of the test sequence.
length = 4
# number of bits used
bits = 1
# num_tests is the number of random examples each network is tested against.
num_tests = 50


def generate_data(seq_length):
    # Adding 2 to bits for writing delim and reading delim
    # also adding 2 to length for delim sequence
    seq = np.zeros([seq_length + 2, bits + 2], dtype=np.float32)
    for idx in range(1, seq_length + 1):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()

    seq[0,0] = 1                # Setting Wrting delim
    seq[seq_length+1,1] = 1     # Setting reading delim
    return seq

# Recall sequence (zeros)
def generate_recall(recall_length):
    seq = np.zeros([recall_length, bits + 2], dtype=np.float32)
    return seq

# Calculate fitness
def calc_error(input, output):
    error = 0.0
    output = list(np.around(np.array(output)))

    for idx in range(bits):
        error += (output[idx] - input[idx + 2]) ** 2
    return error

def eval_genome(genome, config):

    net = neat.nn.RecurrentNetwork.create(genome, config)
    error = 0.0

    for _ in range(num_tests):

        # Create a random sequence with random length to avoid over fitting, and feed it to the network (Write)
        random_length = randint(1, 10)
        sequence = generate_data(random_length)
        recall = generate_recall(random_length)
        inputs = np.concatenate((sequence, recall), axis = 0)
        idx = 0
        read = False
        net.reset()

       
        for input in inputs:
            output = net.activate(input)
            if read:
                idx += 1
                error += calc_error(inputs[idx], output)
            if input[1] == 1 : read = True
            
    total_error = 1.0 - (error / (random_length * num_tests))
    return total_error


def run():
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'CopyTaskOld_config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, 1000)

    # Log statistics.
    stats.save()

    # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    num_correct = 0

    for n in range(num_tests):
        
        print('\nRun {0} output:'.format(n))

        sequence = generate_data(length)
        recall = generate_recall(length)
        inputs = np.concatenate((sequence, recall), axis = 0)
        winner_net.reset()

        print('\tSequence {0}'.format(inputs[:length+2]))

        correct = True
        read = False
        idx = 0

        for input in inputs:
            output = winner_net.activate(input)
            
            if read:
                idx += 1
                error = calc_error(inputs[idx], output)           
                output = list(np.around(np.array(output)))
                print("\texpected {} got {}".format(inputs[idx, 2:bits + 2], output))
                correct = correct and error == 0
            if input[1] == 1 : read = True

        print("OK" if correct else "FAIL")
        num_correct += 1 if correct else 0

    print("{0} of {1} correct {2:.2f}%".format(num_correct, num_tests, 100.0 * num_correct / num_tests))

    node_names = {-1: 'write', -2: 'read', -3: 'sequence', 0: 'output'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    run()
    # random_length = randint(1, 10)
    # sequence = generate_data(random_length)
    # recall = generate_recall(random_length)
    # inputs = np.concatenate((sequence, recall), axis = 0)
    # print(inputs)
    # inputs = [[1., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [0., 0., 1.], [0., 0., 1.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    # outputs = [[1., 0., 0.], [0., 0., 1.], [0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]

    # idx = 0
    # error = 0.0
    # read = False
    
    # for i in range(len(inputs)):
    #     if read:
    #         idx += 1
    #         error += calc_error(inputs[idx], outputs[idx])
    #         print(error)
    #     if inputs[i][1] == 1 : read = True