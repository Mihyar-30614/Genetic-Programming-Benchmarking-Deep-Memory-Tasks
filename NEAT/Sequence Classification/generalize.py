from __future__ import division, print_function

import os
import neat
import numpy as np
import pickle
import random
from sklearn.metrics import accuracy_score

# Data Config
depths = [50,100]       # Number of (1, -1) in a sequence
noise = 10              # Number of Zeros between values
num_tests = 50          # num_tests is the number of random examples each network is tested against.
num_runs = 50           # number of runs

# Results Config
generalize = False
save_log = True
results = []

# Directory of files
local_dir = os.path.dirname(__file__)
rpt_path = os.path.join(local_dir, 'reports/')
champ_path = os.path.join(local_dir, 'champions/')
config_path = os.path.join(local_dir, 'config')


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

if __name__ == "__main__":

    # Load Champion
    print("Loading champions ...")
    champ_name = champ_path + '21_champions_std'
    with open(champ_name, 'rb') as f:
        champions = pickle.load(f)
        print("loaded champions")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    for depth in depths:

        print("Generalizing {} Sequence Length".format(depth))

        for ch in range(num_runs):

            print("Loading Champion {} ....".format(ch+1))
            champion = "champion_" + str(ch+1)
            winner = champions[champion]

            print("Generate Test Dataset ...")
            # Generate Test Dataset
            random_noise = noise

            if generalize:
                random_noise = random.randint(10, 20)
            data_validation = generate_data(depth, random_noise)
            labels_validation = generate_output(data_validation)
            actions_validation = generate_action(data_validation)
            
            print("Begin Testing ....")

            winner_net = neat.nn.RecurrentNetwork.create(winner, config)

            # Evaluate the sum of correctly identified
            predictions, predict_actions = [],[]
            # Evaluate the sum of correctly identified
            for i in range(num_tests):
                data = data_validation[i]
                MEMORY, classification, actions = [], [], []
                counter = 0
                length = len(data)
                for j in range(length):
                    # If stack is empty then 0, else the value on top of stack
                    stack_output = MEMORY[counter - 1] if counter > 0 else 0

                    net_input = [data[j],stack_output]
                    outdata = winner_net.activate(net_input)
                    arg1 = outdata[0]
                    arg2 = outdata[1]
                    arg3 = outdata[2]
                    pos = np.argmax([arg1, arg2, arg3])

                    # Action has been decided
                    temp = 1 if stack_output >= 0 else -1
                    actions.append(pos)
                    if pos == 0:
                        MEMORY.append(data[j])
                        temp = data[j]
                        counter += 1
                    elif pos == 1:
                        if len(MEMORY) > 0:
                            MEMORY.pop()
                        counter -= 1
                        stack_output = MEMORY[counter - 1] if counter > 0 else 0
                        temp = 1 if stack_output >= 0 else -1
                    else:
                        temp = 1 if stack_output >= 0 else -1
                    
                    # Add to classification
                    classification.append(temp)

                predictions.append(classification)
                predict_actions.append(actions)

            # Evaluate predictions
            for i in range(num_tests):
                accuracy = accuracy_score(labels_validation[i], predictions[i])
                results.append(accuracy)
                print("Champion {} Test {} Accuracy: {}".format(ch+1, i+1, accuracy))
            print("==================================================================")
        
        # Save the results
        if save_log:
            with open(rpt_path + 'gen_results_' + str(depth), 'wb') as f:
                pickle.dump(results, f)