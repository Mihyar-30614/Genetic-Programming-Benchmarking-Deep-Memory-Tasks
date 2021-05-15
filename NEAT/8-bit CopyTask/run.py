from __future__ import division, print_function

import os
import visualize
import neat
import numpy as np
import pickle

# length of the test sequence.
length = 5
# number of bits used
bits = 8
# num_tests is the number of random examples each network is tested against.
num_tests = 50


def generate_data(seq_length):
    # Adding 2 to bits for writing delim and reading delim
    # also adding 2 to length for delim sequence
    sequence = np.zeros([seq_length + 2, bits + 2], dtype=np.float32)
    for idx in range(1, seq_length + 1):
        sequence[idx, 2:bits+2] = np.random.rand(bits).round()

    sequence[0, 0] = 1                # Setting Wrting delim
    sequence[seq_length+1, 1] = 1     # Setting reading delim

    recall = np.zeros([seq_length, bits + 2], dtype=np.float32)
    return np.concatenate((sequence, recall), axis=0).tolist()


def generate_output(seq_length):
    data = np.zeros((seq_length*2 + 2, 2), dtype=np.float32)
    data[1:seq_length + 1, 0] = 1
    data[seq_length + 2:seq_length*2 + 2, 1] = 1
    return data.tolist()


def compute_fitness(output_data, expected_data):
    error = 0.0
    output_data = np.around(np.array(output_data)).tolist()
    for indata, outdata in zip(output_data, expected_data):
        if outdata != indata:
            error += 1.0
    fitness = 100.0 - ((error * 100) / len(output_data))
    return fitness


def run(config, winner):
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    num_correct = 0

    for n in range(num_tests):
        print('\nRun {0} output:'.format(n))

        sequence = generate_data(length)
        expected_output = generate_output(length)
        output, MEMORY = [], []
        winner_net.reset()

        print('\tsequence {0}'.format(sequence))
        correct = True
        for I in range(len(sequence)):
            action = "NONE"

            outdata = winner_net.activate(sequence[I])
            stack_push = round(outdata[0])
            stack_pop = round(outdata[1])

            # If Pop and not Push remove the top of stack
            # If Push and not Pop Add sequence to stack
            # Else keep stack as is
            if stack_pop == 1 and stack_push == 0:
                action = "POP"
                if len(MEMORY) > 0:
                    MEMORY.pop()
            elif stack_pop == 0 and stack_push == 1:
                action = "PUSH"
                MEMORY.append(sequence[I][2:])

            # Network output added for fitness evaluate
            output.append(outdata)
            outdata = np.around(np.array(outdata)).tolist()
            print("\texpected {} got {} Action {} Memory {}".format(
                expected_output[I], outdata, action, MEMORY))

        fitness = compute_fitness(output, expected_output)
        correct = correct and fitness == 100
        print("OK" if correct else "FAIL")
        num_correct += 1 if correct else 0

    print("{0} of {1} correct {2:.2f}%".format(
        num_correct, num_tests, 100.0 * num_correct / num_tests))


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
