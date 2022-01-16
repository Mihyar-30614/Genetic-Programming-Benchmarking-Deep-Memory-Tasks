from turtle import color
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import csv

local_dir = os.path.dirname(__file__)
axis_x_250 = list(range(0, 251))
axis_x_500 = list(range(0, 500))
axis_x_501 = list(range(0, 501))
y_label = "Success Percentage"
x_label = "Training Generations"
legend_loc = "lower right"
color1 = "green"
color2 = "maroon"

'''
    Load Data
'''

'''
    Copy Task
'''
# 8-bit Report DEAP
DEAP_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/8-bit-report/8-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_Standard.append(info)

# 8-bit Report NEAT
NEAT_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/8-bit CopyTask/8-bit-report/fitness_history' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            info.append(float(*row))
    NEAT_Standard.append(info)

# DEAP Modified Task
mod_info = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/8-bit-report/8-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    mod_info.append(info)

'''
    Sequence Recall
'''
# 5-deep Report DEAP
DEAP_5_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/5-deep-report/5-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_5_Standard.append(info)

# 5-deep Report NEAT
NEAT_5_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Recall/5-deep-report/5-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
        info.extend([info[-1] for _ in range(500 - len(info))])
    NEAT_5_Standard.append(info)

'''
    Sequence Classification
'''
# 5-deep Report DEAP
DEAP_5_Standard_2 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/5-deep-report/5-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_5_Standard_2.append(info)

# 5-deep Report NEAT
NEAT_5_Standard_2 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Classification/5-deep-report/5-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
    NEAT_5_Standard_2.append(info)

# 15-deep Report DEAP
DEAP_15_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/15-deep-report/15-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_15_Standard.append(info)

# 21-deep Report DEAP
DEAP_21_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/21-deep-report/21-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_21_Standard.append(info)

# 15-deep Report DEAP Logical
DEAP_15_Logical = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/15-deep-report/15-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_15_Logical.append(info)

# 21-deep Report DEAP Logical
DEAP_21_Logical = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/21-deep-report/21-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_21_Logical.append(info)

'''
    Calculate Mean and Standard Deviation
'''
DEAP_standard_mean = np.mean(DEAP_Standard, axis=0)
DEAP_standard_std = np.std(DEAP_Standard, axis=0)
DEAP_standard_upperlimit = np.clip(np.add(DEAP_standard_mean, DEAP_standard_std), a_min=0, a_max=100)
DEAP_standard_lowerlimit = np.clip(np.subtract(DEAP_standard_mean, DEAP_standard_std), a_min=0, a_max=100)

NEAT_standard_mean = np.mean(NEAT_Standard, axis=0)
NEAT_standard_std = np.std(NEAT_Standard, axis=0)
NEAT_standard_upperlimit = np.clip(np.add(NEAT_standard_mean, NEAT_standard_std), a_min=0, a_max=100)
NEAT_standard_lowerlimit = np.clip(np.subtract(NEAT_standard_mean, NEAT_standard_std), a_min=0, a_max=100)

mod_mean = np.mean(mod_info, axis=0)
mod_std = np.std(mod_info, axis=0)
DEAP_mod_upperlimit = np.clip(np.add(mod_mean, mod_std), a_min=0, a_max=100)
DEAP_mod_lowerlimit = np.clip(np.subtract(mod_mean, mod_std), a_min=0, a_max=100)

DEAP_5_Standard_mean = np.mean(DEAP_5_Standard, axis=0)
DEAP_5_Standard_std = np.std(DEAP_5_Standard, axis=0)
DEAP_5_Standard_upperlimit = np.clip(np.add(DEAP_5_Standard_mean, DEAP_5_Standard_std), a_min=0, a_max=100)
DEAP_5_Standard_lowerlimit = np.clip(np.subtract(DEAP_5_Standard_mean, DEAP_5_Standard_std), a_min=0, a_max=100)

NEAT_5_Standard_mean = np.mean(NEAT_5_Standard, axis=0)
NEAT_5_Standard_std = np.std(NEAT_5_Standard, axis=0)
NEAT_5_Standard_upperlimit = np.clip(np.add(NEAT_5_Standard_mean, NEAT_5_Standard_std), a_min=0, a_max=100)
NEAT_5_Standard_lowerlimit = np.clip(np.subtract(NEAT_5_Standard_mean, NEAT_5_Standard_std), a_min=0, a_max=100)

DEAP_5_Standard_2_mean = np.mean(DEAP_5_Standard_2, axis=0)
DEAP_5_Standard_2_std = np.std(DEAP_5_Standard_2, axis=0)
DEAP_5_Standard_2_upperlimit = np.clip(np.add(DEAP_5_Standard_2_mean, DEAP_5_Standard_2_std), a_min=0, a_max=100)
DEAP_5_Standard_2_lowerlimit = np.clip(np.subtract(DEAP_5_Standard_2_mean, DEAP_5_Standard_2_std), a_min=0, a_max=100)

NEAT_5_Standard_2_mean = np.mean(NEAT_5_Standard_2, axis=0)
NEAT_5_Standard_2_std = np.std(NEAT_5_Standard_2, axis=0)
NEAT_5_Standard_2_upperlimit = np.clip(np.add(NEAT_5_Standard_2_mean, NEAT_5_Standard_2_std), a_min=0, a_max=100)
NEAT_5_Standard_2_lowerlimit = np.clip(np.subtract(NEAT_5_Standard_2_mean, NEAT_5_Standard_2_std), a_min=0, a_max=100)

DEAP_15_Standard_mean = np.mean(DEAP_15_Standard, axis=0)
DEAP_15_Standard_std = np.std(DEAP_15_Standard, axis=0)
DEAP_15_Standard_upperlimit = np.clip(np.add(DEAP_15_Standard_mean, DEAP_15_Standard_std), a_min=0, a_max=100)
DEAP_15_Standard_lowerlimit = np.clip(np.subtract(DEAP_15_Standard_mean, DEAP_15_Standard_std), a_min=0, a_max=100)

DEAP_21_Standard_mean = np.mean(DEAP_21_Standard, axis=0)
DEAP_21_Standard_std = np.std(DEAP_21_Standard, axis=0)
DEAP_21_Standard_upperlimit = np.clip(np.add(DEAP_21_Standard_mean, DEAP_21_Standard_std), a_min=0, a_max=100)
DEAP_21_Standard_lowerlimit = np.clip(np.subtract(DEAP_21_Standard_mean, DEAP_21_Standard_std), a_min=0, a_max=100)

DEAP_15_Logical_mean = np.mean(DEAP_15_Logical, axis=0)
DEAP_15_Logical_std = np.std(DEAP_15_Logical, axis=0)
DEAP_15_Logical_upperlimit = np.clip(np.add(DEAP_15_Logical_mean, DEAP_15_Logical_std), a_min=0, a_max=100)
DEAP_15_Logical_lowerlimit = np.clip(np.subtract(DEAP_15_Logical_mean, DEAP_15_Logical_std), a_min=0, a_max=100)

DEAP_21_Logical_mean = np.mean(DEAP_21_Logical, axis=0)
DEAP_21_Logical_std = np.std(DEAP_21_Logical, axis=0)
DEAP_21_Logical_upperlimit = np.clip(np.add(DEAP_21_Logical_mean, DEAP_21_Logical_std), a_min=0, a_max=100)
DEAP_21_Logical_lowerlimit = np.clip(np.subtract(DEAP_21_Logical_mean, DEAP_21_Logical_std), a_min=0, a_max=100)

'''
    Plot Results
'''
# Copy Task
plt.figure(1)
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="GP", color=color1)
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, color=color1, alpha=.3)
plt.plot(axis_x_500, NEAT_standard_mean, linewidth=1, label="NEAT", color=color2)
plt.fill_between(axis_x_500, NEAT_standard_lowerlimit, NEAT_standard_upperlimit, color=color2, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Gecco Paper/copy_task.png", bbox_inches='tight')

# Seq Recall
plt.figure(2)
plt.plot(axis_x_250, DEAP_5_Standard_mean, linewidth=1, label="5-depth GP", color=color1)
plt.fill_between(axis_x_250, DEAP_5_Standard_lowerlimit, DEAP_5_Standard_upperlimit, color=color1, alpha=.3)
plt.plot(axis_x_500, NEAT_5_Standard_mean, linewidth=1, label="5-depth NEAT", color=color2)
plt.fill_between(axis_x_500, NEAT_5_Standard_lowerlimit, NEAT_5_Standard_upperlimit, color=color2, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Gecco Paper/seq_recall_5.png", bbox_inches='tight')

plt.figure(3)
plt.plot(axis_x_250, DEAP_5_Standard_2_mean, linewidth=1, label="5-depth GP", color=color1)
plt.fill_between(axis_x_250, DEAP_5_Standard_2_lowerlimit, DEAP_5_Standard_2_upperlimit, color=color1, alpha=.3)
plt.plot(axis_x_500, NEAT_5_Standard_2_mean, linewidth=1, label="5-depth NEAT", color=color2)
plt.fill_between(axis_x_500, NEAT_5_Standard_2_lowerlimit, NEAT_5_Standard_2_upperlimit, color=color2, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Gecco Paper/seq_class_5.png", bbox_inches='tight')

plt.figure(4)
plt.plot(axis_x_250, DEAP_15_Standard_mean, linewidth=1, label="15-depth Div", color=color1)
plt.fill_between(axis_x_250, DEAP_15_Standard_lowerlimit, DEAP_15_Standard_upperlimit, color=color1, alpha=.3)
plt.plot(axis_x_250, DEAP_15_Logical_mean, linewidth=1, label="15-depth Full", color=color2)
plt.fill_between(axis_x_250, DEAP_15_Logical_lowerlimit, DEAP_15_Logical_upperlimit, color=color2, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Gecco Paper/seq_class_15.png", bbox_inches='tight')

plt.figure(5)
plt.plot(axis_x_250, DEAP_21_Standard_mean, linewidth=1, label="21-depth Div", color=color1)
plt.fill_between(axis_x_250, DEAP_21_Standard_lowerlimit, DEAP_21_Standard_upperlimit, color=color1, alpha=.3)
plt.plot(axis_x_250, DEAP_21_Logical_mean, linewidth=1, label="21-depth Full", color=color2)
plt.fill_between(axis_x_250, DEAP_21_Logical_lowerlimit, DEAP_21_Logical_upperlimit, color=color2, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Gecco Paper/seq_class_21.png", bbox_inches='tight')

# Original vs Modified
plt.figure(6)
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="Original", color=color1)
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, color=color1, alpha=.3)
plt.plot(axis_x_250, mod_mean, linewidth=1, label="Modified", color=color2)
plt.fill_between(axis_x_250, DEAP_mod_lowerlimit, DEAP_mod_upperlimit, color=color2, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Gecco Paper/copy_task_modified.png", bbox_inches='tight')

plt.legend(loc=legend_loc)
plt.show()