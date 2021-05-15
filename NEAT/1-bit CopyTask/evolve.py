from __future__ import division, print_function

import multiprocessing
import os
import visualize
import neat
import numpy as np
import random
import pickle

# length of the test sequence.
length = 3
# number of bits used
bits = 1
# num_tests is the number of random examples each network is tested against.
num_tests = 50
num_generations = 1000
generalize = True


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


def network_simulator(input_data, mode):
    global switch
    if mode == "PERFECT":
        retval = [0.0, 0.0]
        if input_data[0] == 1 and input_data[1] == 0:
            switch = "PUSH"
        elif input_data[0] == 0 and input_data[1] == 1:
            switch = "POP"
        else:
            if switch == "PUSH":
                retval = [1.0, 0.0]
            elif switch == "POP":
                retval = [0.0, 1.0]
            else:
                raise ValueError("Switch is NONE")
    elif mode == "RANDOM":
        delim1 = random.choice((0.0, 1.0))
        delim2 = random.choice((0.0, 1.0))
        retval = [delim1, delim2]
    else:
        raise ValueError("Unknown Mode")
    return retval


def run_simulator():
    total_fitness = 0.0
    global length
    for _ in range(num_tests):
        # Create a random sequence, and feed it to the network (Write)
        if generalize:
            length = random.randint(1, 3)
        sequence = generate_data(length)
        expected_output = generate_output(length)
        output, MEMORY = [], []

        print('\tSequence {}'.format(sequence))
        correct = True
        global switch
        switch = "NONE"

        for I in range(len(sequence)):
            action = "NONE"

            outdata = network_simulator(sequence[I], "RANDOM")
            stack_push = round(outdata[0])
            stack_pop = round(outdata[1])

            if stack_pop == 1 and stack_push == 0:
                action = "POP"
                if len(MEMORY) > 0:
                    MEMORY.pop()
            elif stack_pop == 0 and stack_push == 1:
                action = "PUSH"
                MEMORY.append(sequence[I][2:])

            # Network output added for fitness evaluate
            output.append(outdata)

            print("\texpected {} got {} Action {} Memory {}".format(
                expected_output[I], outdata, action, MEMORY))

        fitness = compute_fitness(output, expected_output)
        total_fitness += fitness
        correct = correct and fitness == 100

        print("Fitness: {}".format(fitness))
        print("OK" if correct else "FAIL")

    print("Total Fitness: {}".format(total_fitness/num_tests))


def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    total_fitness = 0.0
    global length
    for _ in range(num_tests):
        # Create a random sequence, and feed it to the network (Write)
        if generalize:
            length = random.randint(1, 3)
        sequence = generate_data(length)
        expected_output = generate_output(length)
        output, MEMORY = [], []
        net.reset()

        for I in range(len(sequence)):

            outdata = net.activate(sequence[I])
            stack_push = round(outdata[0])
            stack_pop = round(outdata[1])

            if stack_pop == 1 and stack_push == 0:
                if len(MEMORY) > 0:
                    MEMORY.pop()
            elif stack_pop == 0 and stack_push == 1:
                MEMORY.append(sequence[I][2:])

            # Network output added for fitness evaluate
            output.append(outdata)

        fitness = compute_fitness(output, expected_output)
        total_fitness += fitness

    return total_fitness/num_tests


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
    winner = pop.run(pe.evaluate, num_generations)

    # Log statistics.
    stats.save()

    # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the winner
    with open('champion-gnome', 'wb') as f:
        pickle.dump(winner, f)


if __name__ == "__main__":

    # Run Training
    run()
    # run_simulator()
