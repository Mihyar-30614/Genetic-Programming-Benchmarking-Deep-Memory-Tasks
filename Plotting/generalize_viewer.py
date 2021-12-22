import pickle
import os
import numpy as np

local_dir = os.path.dirname(__file__)

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

path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit-vector/champion/results_50')
with open(path, 'rb') as f:
    Values4_50 = pickle.load(f)

path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit-vector/champion/results_100')
with open(path, 'rb') as f:
    Values4_100 = pickle.load(f)


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

Values4_50_mean = np.mean(Values4_50, axis=0)
Values4_50_std = np.std(Values4_50, axis=0)
Values4_100_mean = np.mean(Values4_100, axis=0)
Values4_100_std = np.std(Values4_100, axis=0)

'''
    View Last item as result
'''
print("=================================")
print("Copy Task length 50")
print("Mean: {:.2f}%".format(Values1_50_mean * 100))
print("STD: {:.2f}".format(Values1_50_std * 100))
print("Copy Task length 100")
print("Mean: {:.2f}%".format(Values1_100_mean * 100))
print("STD: {:.2f}".format(Values1_100_std * 100))

print("=================================")
print("Copy Task Vector length 50")
print("Mean: {:.2f}%".format(Values4_50_mean * 100))
print("STD: {:.2f}".format(Values4_50_std * 100))
print("Copy Task Vector length 100")
print("Mean: {:.2f}%".format(Values4_100_mean * 100))
print("STD: {:.2f}".format(Values4_100_std * 100))

print("=================================")
print("Sequence Classification length 50")
print("Mean: {:.2f}%".format(Values2_50_mean * 100))
print("STD: {:.2f}".format(Values2_50_std * 100))
print("Sequence Classification length 100")
print("Mean: {:.2f}%".format(Values2_100_mean * 100))
print("STD: {:.2f}".format(Values2_100_std * 100))
print("=================================")

print("Sequence Recall length 50")
print("Mean: {:.2f}%".format(Values3_50_mean * 100))
print("STD: {:.2f}".format(Values3_50_std * 100))
print("Sequence Recall length 100")
print("Mean: {:.2f}%".format(Values3_100_mean * 100))
print("STD: {:.2f}".format(Values3_100_std * 100))
print("=================================")