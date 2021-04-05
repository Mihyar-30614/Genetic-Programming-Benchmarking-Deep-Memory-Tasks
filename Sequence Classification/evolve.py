from __future__ import division, print_function
import neat
import visualize
import numpy as np
import random
import multiprocessing
import os

# Number of (1, -1) in a sequence
depth = 4
# Number of Zeros between values
noise = 10
# num_tests is the number of random examples each network is tested against.
num_tests = 50
num_generations = 1000
MEMORY = []

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
    return 1.0 - (error * 100) / len(output_data)

def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    error = 0.0
    for _ in range(num_tests):
        # Create a random sequence, and feed it to the network (Write)
        random_noise = random.randint(10, 20)
        sequence = generate_data(depth, noise)
        expected_output = generate_output(sequence)
        output = []
        stack_output = 0
        net.reset()

        for seq in sequence:
            temp = net.activate(seq, stack_output)
            stack_pop = round(temp[0])
            stack_push = round(temp[1])
            outdata = temp[2]
            outdata = -1.0 if outdata < 0.5 else 1.0
            output.append(outdata)
            
        error += compute_fitness(output, expected_output)
            
    total_error = error / num_tests
    return total_error

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
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    num_correct = 0


if __name__ == "__main__":
    # data = generate_data(depth, noise)
    # output = generate_output(data)
    # fitness = compute_fitness(output, output)
    # print("Data:\n", data)
    # print("output:\n", output)
    # print("fitness:\n", fitness)
    run()