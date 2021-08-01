'''
    Verify the results from DEAP
'''

import random
import operator
import numpy as np
from sklearn.metrics import accuracy_score

depth = 100
corridor_length = 10
num_tests = 50
generalize = True

'''
Problem setup
'''

# Define a protected division function
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

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
        for i in range(len(instruction)):
            # 0 = PUSH, 1 = POP HEAD, 2 = NOTHING, 3 = POP TAIL
            if instruction[i] == 1:
                output.append(0)
            elif instruction[i] == 0 and data[i] == 0:
                output.append(1)
            else:
                output.append(2)
        retval.append(output)
    return retval

data_validation = generate_data(depth, corridor_length)
actions_validation = generate_action(data_validation)


def tree1(ARG0, ARG1):
    return operator.add(ARG0, ARG0)

def tree2(ARG0, ARG1):
    return protected_div(ARG0, operator.sub(operator.add(ARG1, ARG0), operator.add(ARG1, ARG1)))

def tree3(ARG0, ARG1):
    return operator.add(ARG1, ARG1)

def tree4(ARG0, ARG1):
    return protected_div(operator.sub(ARG1, ARG0), operator.sub(operator.sub(ARG0, ARG0), ARG1))


if __name__ == "__main__":

    '''
    Running Test on unseen data and checking results
    '''
    print("\n==================")
    print("Begin Testing ....")
    print("==================\n")

    # Evaluate the sum of correctly identified
    predict_actions = []
    # Evaluate the sum of correctly identified
    for i in range(num_tests):
        instructions, data, actions = data_validation[i][0], data_validation[i][1], []
        length = len(data)

        for j in range(length):
            arg1 = tree1(instructions[j], data[j])
            arg2 = tree2(instructions[j], data[j])
            arg3 = tree3(instructions[j], data[j])
            arg4 = tree4(instructions[j], data[j])
            pos = np.argmax([arg1, arg2, arg3, arg4])
            actions.append(pos)

        predict_actions.append(actions)

    # Evaluate predictions
    total_accuracy = 0
    for i in range(num_tests):
        print("instructrions: \n{}".format(data_validation[i][0]))
        print("data: \n{}".format(data_validation[i][1]))
        print("Prediction Actions: \n{}".format(predict_actions[i]))
        print("Actions: \n{}".format(actions_validation[i]))
        accuracy = accuracy_score(actions_validation[i], predict_actions[i])
        print("Accuracy: {}".format(accuracy))
        print("==================================================================")
        total_accuracy += accuracy
    
    print("Total Accuracy: {}".format(total_accuracy/num_tests))

