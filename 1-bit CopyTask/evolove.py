from __future__ import division, print_function
from random import randint

import multiprocessing
import os
import visualize
import neat
import numpy as np
import math

# length of the test sequence.
length = 3
# number of bits used
bits = 1
# num_tests is the number of random examples each network is tested against.
num_tests = 50
memory = []


def generate_data(seq_length):
    # Adding 2 to bits for writing delim and reading delim
    # also adding 2 to length for delim sequence
    data = np.zeros([seq_length + 2, bits + 2], dtype=np.float32)
    for idx in range(1, seq_length + 1):
        data[idx, 2:bits+2] = np.random.rand(bits).round()

    data[0,0] = 1                # Setting Wrting delim
    data[seq_length+1,1] = 1     # Setting reading delim
    return data

def generate_recall(recall_length):
    return np.zeros([recall_length, bits + 2], dtype=np.float32)

def generate_output(seq_length):
    data = np.zeros((seq_length*2 + 2, 2), dtype=np.float32)
    data[1:seq_length +1, 1] = 1
    data[seq_length +2:seq_length*2 +2, 0] = 1
    return data

def calc_error(input, output):
    error = 0.0
    output = list(np.around(np.array(output)))

    for idx in range(bits):
        # ignore delim part of the sequence
        if input[0] != 1 and input[1] != 1:
            error += (output[idx] - input[idx + 2]) ** 2
    return error

def calc_error_test(output, expected_output):
    error = 0.0
    output = list(np.around(np.array(output)))
    for idx in range(2):
        error += (output[idx] - expected_output[idx]) ** 2
    return error

def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    error = 0.0
    for _ in range(num_tests):
        # Create a random sequence, and feed it to the network (Write)
        random_length = randint(1, 3)
        sequence = generate_data(random_length)
        recall = generate_recall(random_length)
        expected_output = generate_output(random_length)
        inputs = np.concatenate((sequence, recall), axis = 0)
        net.reset()

        for xi, xo in zip(inputs, expected_output):
            output = net.activate(xi)
            error += calc_error_test(output, xo)
            
    total_error = 1.0 - (error / (random_length * num_tests))
    return total_error


def run():
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

        seq = generate_data(length)
        recall = generate_recall(length)
        inputs = np.concatenate((seq, recall), axis = 0)
        expected_output = generate_output(length)
        winner_net.reset()

        print('\tsequence {0}'.format(inputs))
        correct = True
        for xi, xo in zip(inputs, expected_output):
            output = winner_net.activate(xi)

            error = calc_error_test(output, xo)
            output = list(np.around(np.array(output)))
            print("\texpected {} got {}".format(xo, output))
            correct = correct and error == 0
        print("OK" if correct else "FAIL")
        num_correct += 1 if correct else 0

    print("{0} of {1} correct {2:.2f}%".format(num_correct, num_tests, 100.0 * num_correct / num_tests))

    node_names = {-1: 'write', -2: 'read', -3: 'sequence', 0: 'output'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    run()
