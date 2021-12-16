import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import csv

local_dir = os.path.dirname(__file__)
axis_x_250 = list(range(0, 251))
axis_x_500 = list(range(0, 500))

'''
    Load Data
'''
path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/champion/results_50')
with open(path, 'rb') as f:
    Values1_50 = pickle.load(f)

path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/champion/results_100')
with open(path, 'rb') as f:
    Values1_100 = pickle.load(f)

path = os.path.join(local_dir, '../DEAP/Sequence Classification/champion/results_50')
with open(path, 'rb') as f:
    Values2_50 = pickle.load(f)

path = os.path.join(local_dir, '../DEAP/Sequence Classification/champion/results_100')
with open(path, 'rb') as f:
    Values2_100 = pickle.load(f)

path = os.path.join(local_dir, '../DEAP/Sequence Recall/champion/results_50')
with open(path, 'rb') as f:
    Values3_50 = pickle.load(f)

path = os.path.join(local_dir, '../DEAP/Sequence Recall/champion/results_100')
with open(path, 'rb') as f:
    Values3_100 = pickle.load(f)


'''
    Calculate Mean and Standard Deviation
'''
Values1_50_mean = np.mean(Values1_50, axis=0)
Values1_50_std = np.std(Values1_50, axis=0)
Values1_100_mean = np.mean(Values1_100, axis=0)
Values1_100_std = np.std(Values1_100, axis=0)

Values2_50_mean = np.mean(Values2_50, axis=0)
Values2_50_std = np.std(Values2_50, axis=0)
Values2_100_mean = np.mean(Values2_100, axis=0)
Values2_100_std = np.std(Values2_100, axis=0)

Values3_50_mean = np.mean(Values3_50, axis=0)
Values3_50_std = np.std(Values3_50, axis=0)
Values3_100_mean = np.mean(Values3_100, axis=0)
Values3_100_std = np.std(Values3_100, axis=0)

'''
    View Last item as result
'''
print("Copy Task length 50")
print("Mean: {}".format(Values1_50_mean))
print("STD: {}".format(Values1_50_std))
print("Copy Task length 100")
print("Mean: {}".format(Values1_100_mean))
print("STD: {}".format(Values1_100_std))

print("Sequence Classification length 50")
print("Mean: {}".format(Values2_50_mean))
print("STD: {}".format(Values2_50_std))
print("Sequence Classification length 100")
print("Mean: {}".format(Values2_100_mean))
print("STD: {}".format(Values2_100_std))

print("Sequence Recall length 50")
print("Mean: {}".format(Values3_50_mean))
print("STD: {}".format(Values3_50_std))
print("Sequence Recall length 100")
print("Mean: {}".format(Values3_100_mean))
print("STD: {}".format(Values3_100_std))