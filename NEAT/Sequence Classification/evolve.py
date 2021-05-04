"""
This is an example of sequence classification using NEAT-Python.

Example Input:
    sequence = [1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0]
    Stack_output = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0]
    
Example Output:
    stack_push = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    stack_pop = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    classification = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
"""

from __future__ import division, print_function
import neat
import visualize
import numpy as np
import random
import multiprocessing
import os
import pickle

# Number of (1, -1) in a sequence
depth = 4
# Number of Zeros between values
noise = 10
# num_tests is the number of random examples each network is tested against.
num_tests = 50
num_generations = 1000
gneralize = False


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

def network_simulator(sequence, stack, mode):
    
    if mode == "PERFECT":
        # produce perfect output
        push = 0
        pop = 0
        classification = 1 if stack >= 0 else -1
        
        if sequence != 0 and stack == 0:
        # if sequence not zero and stack is zero then push
            push = 1
            pop = 0
            classification = sequence
        elif sequence != 0 and stack != 0:
        # if sequence not zero and stack is not zero then 
        # (if sequence equal to stack then push otherwise pop)
            if sequence == stack:
                push = 1
                pop = 0
            else:
                push = 0
                pop = 1  
    else:
        # produce random output
        push = random.choice([0,1])
        pop = random.choice([0,1])
        classification = random.choice([-1,1])
        
    return [push, pop, classification]


def run_simulator():
    total_fitness = 0.0    
    for _ in range(num_tests):
        # Create a random sequence, and feed it to the network (Write)
        random_noise = random.randint(10, 20)
        sequence = generate_data(depth, random_noise)
        expected_output = generate_output(sequence)
        classification = []
        MEMORY = []
        counter = 0

        print('\tsequence {0}'.format(sequence))
        correct = True
        for I in range(len(sequence)):
            # If stack is empty then 0, else the value on top of stack
            stack_output = MEMORY[counter-1] if counter > 0 else 0
            action = "NONE"

            temp = network_simulator(sequence[I], stack_output, "PERFECT")
            stack_push = round(temp[0])
            stack_pop = round(temp[1])

            if stack_pop == 1 and stack_push == 0:
                counter -= 1
                action = "POP"
                MEMORY.pop()
            elif stack_pop == 0 and stack_push == 1:
                counter += 1
                action = "PUSH"
                MEMORY.append(sequence[I])

            # Network output added for fitness evaluate
            outdata = temp[2]
            outdata = -1.0 if outdata < 0.5 else 1.0
            classification.append(outdata)
            
            print("\texpected {} got {} Action {} Memory {}".format(expected_output[I], outdata, action, MEMORY))
        fitness = compute_fitness(classification, expected_output)
        total_fitness += fitness
        correct = correct and fitness == 100
        print("OK" if correct else "FAIL")

    return total_fitness / num_tests

def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    fitness = 0.0
    for _ in range(num_tests):
        # Create a random sequence, and feed it to the network (Write)
        random_noise  = noise
        if gneralize:
            random_noise = random.randint(10, 20)
        sequence = generate_data(depth, random_noise)
        expected_output = generate_output(sequence)
        classification = []
        MEMORY = []
        counter = 0
        net.reset()

        for seq in sequence:
            # If stack is empty then 0, else the value on top of stack
            stack_output = MEMORY[counter -1] if counter > 0 else 0

            temp = net.activate([seq, stack_output])
            stack_push = round(temp[0])
            stack_pop = round(temp[1])

            # If Pop and not Push remove the top of stack
            # If Push and not Pop Add sequence to stack
            # Else keep stack as is
            if stack_pop == 1 and stack_push == 0:
                if len(MEMORY) > 0:
                    counter -= 1
                    MEMORY.pop()
            elif stack_pop == 0 and stack_push == 1:
                counter += 1
                MEMORY.append(seq)

            # Network output added for fitness evaluate
            outdata = temp[2]
            outdata = -1.0 if outdata < 0.5 else 1.0
            classification.append(outdata)
            
        fitness += compute_fitness(classification, expected_output)
            
    total_fitness = fitness / num_tests
    return total_fitness

def run():
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, num_generations)

    # Log statistics.
    stats.save()

    # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')

    # Save the winner
    with open('champion-gnome', 'wb') as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    
    # Run Training
    run()
    # run_simulator()