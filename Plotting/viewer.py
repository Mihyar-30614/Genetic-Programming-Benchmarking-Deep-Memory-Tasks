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
Values = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/21-deep-report/21-progress_report_0.125_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    Values.append(info)


# for i in range(1,21):
#     path = os.path.join(local_dir, '../NEAT/Sequence Classification/21-deep-report/21-progress_report' + str(i) + '.csv')
#     info = []
#     with open(path, newline='\n') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             fitness = row[0].split()
#             info.append(float(fitness[0]))
#     Values.append(info)


'''
    Calculate Mean and Standard Deviation
'''
Values_mean = np.mean(Values, axis=0)
Values_std = np.std(Values, axis=0)

'''
    View Last item as result
'''
print("Mean: {}".format(Values_mean[-1]))
print("STD: {}".format(Values_std[-1]))