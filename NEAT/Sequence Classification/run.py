from __future__ import division, print_function
import neat
import visualize
import numpy as np
import random
import os
import pickle

# Number of (1, -1) in a sequence
depth = 4
# Number of Zeros between values
noise = 10
# num_tests is the number of random examples each network is tested against.
num_tests = 50

def generate_data(depth, noise):
    sequence = []
    sequence.append(random.choice((-1.0, 1.0)))
    for _ in range(depth - 1):
        sequence.extend([0 for _ in range(noise)])
        sequence.append(random.choice((-1.0, 1.0)))
    return sequence

def generate_output(data):
    retval = []
    counter = 0
    for el in data:
        counter += el
        retval.append(-1 if counter < 0 else 1)
    return retval

def compute_fitness(output_data, expected_data):
    error = 0.0
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

        sequence = generate_data(depth, noise)
        expected_output = generate_output(sequence)
        classification = []
        MEMORY = []
        counter = 0
        winner_net.reset()

        print('\tsequence {0}'.format(sequence))
        correct = True
        for I in range(len(sequence)):
            # If stack is empty then 0, else the value on top of stack
            stack_output = MEMORY[counter -1] if counter > 0 else 0
            action = "NONE"

            outdata = winner_net.activate([sequence[I], stack_output])
            stack_push = round(outdata[0])
            stack_pop = round(outdata[1])

            # If Pop and not Push remove the top of stack
            # If Push and not Pop Add sequence to stack
            # Else keep stack as is
            if stack_pop == 1 and stack_push == 0:
                if len(MEMORY) > 0:
                    counter -= 1
                    action = "POP"
                    MEMORY.pop()
            elif stack_pop == 0 and stack_push == 1:
                counter += 1
                action = "PUSH"
                MEMORY.append(sequence[I])

            # Network output added for fitness evaluate
            output = outdata[2]
            output = -1.0 if output < 0.5 else 1.0
            classification.append(output)

            print("\texpected {} got {} Action {} Memory {}".format(expected_output[I], output, action, MEMORY))
        
        fitness = compute_fitness(classification, expected_output)
        correct = correct and fitness == 100
        print("OK" if correct else "FAIL")
        num_correct += 1 if correct else 0

    print("{0} of {1} correct {2:.2f}%".format(num_correct, num_tests, 100.0 * num_correct / num_tests))


if __name__ == "__main__":

    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
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