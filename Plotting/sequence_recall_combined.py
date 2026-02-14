import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import csv

local_dir = os.path.dirname(__file__)
axis_x_250 = list(range(251))
axis_x_500 = list(range(500))
y_label = "Success Percentage"
x_label = "Training Generations"
legend_loc = "lower right"

'''
    Load Data
'''
# 4-deep Report DEAP
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/4_report_std')
with open(path, 'rb') as f:
    DEAP_4_Standard = list(pickle.load(f).values())

# 5-deep Report DEAP
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/5_report_std')
with open(path, 'rb') as f:
    DEAP_5_Standard = list(pickle.load(f).values())

# 6-deep Report DEAP
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/6_report_std')
with open(path, 'rb') as f:
    DEAP_6_Standard = list(pickle.load(f).values())

# 15-deep Report DEAP
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/15_report_std')
with open(path, 'rb') as f:
    DEAP_15_Standard = list(pickle.load(f).values())

# 21-deep Report DEAP
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/21_report_std')
with open(path, 'rb') as f:
    DEAP_21_Standard = list(pickle.load(f).values())

# 4-deep Report DEAP Logical
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/4_report_log')
with open(path, 'rb') as f:
    DEAP_4_Logical = list(pickle.load(f).values())

# 5-deep Report DEAP Logical
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/5_report_log')
with open(path, 'rb') as f:
    DEAP_5_Logical = list(pickle.load(f).values())

# 6-deep Report DEAP Logical
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/6_report_log')
with open(path, 'rb') as f:
    DEAP_6_Logical = list(pickle.load(f).values())

# 15-deep Report DEAP Logical
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/15_report_log')
with open(path, 'rb') as f:
    DEAP_15_Logical = list(pickle.load(f).values())

# 21-deep Report DEAP Logical
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/21_report_log')
with open(path, 'rb') as f:
    DEAP_21_Logical = list(pickle.load(f).values())

# 4-deep Report DEAP Multiplication
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/4_report_mul')
with open(path, 'rb') as f:
    DEAP_4_Multiplication = list(pickle.load(f).values())

# 5-deep Report DEAP Multiplication
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/5_report_mul')
with open(path, 'rb') as f:
    DEAP_5_Multiplication = list(pickle.load(f).values())

# 4-deep Report DEAP Modified
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/4_report_mod')
with open(path, 'rb') as f:
    DEAP_4_Modified = list(pickle.load(f).values())

# 5-deep Report DEAP Modified
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/5_report_mod')
with open(path, 'rb') as f:
    DEAP_5_Modified = list(pickle.load(f).values())

# 6-deep Report DEAP Modified
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/6_report_mod')
with open(path, 'rb') as f:
    DEAP_6_Modified = list(pickle.load(f).values())

# 15-deep Report DEAP Modified
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/15_report_mod')
with open(path, 'rb') as f:
    DEAP_15_Modified = list(pickle.load(f).values())

# 21-deep Report DEAP Modified
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/21_report_mod')
with open(path, 'rb') as f:
    DEAP_21_Modified = list(pickle.load(f).values())

# 4-deep Report NEAT
NEAT_4_Standard = []
path = os.path.join(local_dir, '../NEAT/Sequence Recall/reports/4_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    NEAT_4_Standard.append(info)

# 5-deep Report NEAT
NEAT_5_Standard = []
path = os.path.join(local_dir, '../NEAT/Sequence Recall/reports/5_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    NEAT_5_Standard.append(info)

# 6-deep Report NEAT
NEAT_6_Standard = []
path = os.path.join(local_dir, '../NEAT/Sequence Recall/reports/6_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    NEAT_6_Standard.append(info)

# 15-deep Report NEAT
NEAT_15_Standard = []
path = os.path.join(local_dir, '../NEAT/Sequence Recall/reports/15_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    NEAT_15_Standard.append(info)

# 21-deep Report NEAT
NEAT_21_Standard = []
path = os.path.join(local_dir, '../NEAT/Sequence Recall/reports/21_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    NEAT_21_Standard.append(info)

'''
    Calculate Mean and Standard Deviation
'''

DEAP_4_Standard_mean = np.mean(DEAP_4_Standard, axis=0)
DEAP_4_Standard_std = np.std(DEAP_4_Standard, axis=0)
DEAP_4_Standard_upperlimit = np.clip(np.add(DEAP_4_Standard_mean, DEAP_4_Standard_std), a_min=0, a_max=100)
DEAP_4_Standard_lowerlimit = np.clip(np.subtract(DEAP_4_Standard_mean, DEAP_4_Standard_std), a_min=0, a_max=100)

NEAT_4_Standard_mean = np.mean(NEAT_4_Standard, axis=0)
NEAT_4_Standard_std = np.std(NEAT_4_Standard, axis=0)
NEAT_4_Standard_upperlimit = np.clip(np.add(NEAT_4_Standard_mean, NEAT_4_Standard_std), a_min=0, a_max=100)
NEAT_4_Standard_lowerlimit = np.clip(np.subtract(NEAT_4_Standard_mean, NEAT_4_Standard_std), a_min=0, a_max=100)

DEAP_5_Standard_mean = np.mean(DEAP_5_Standard, axis=0)
DEAP_5_Standard_std = np.std(DEAP_5_Standard, axis=0)
DEAP_5_Standard_upperlimit = np.clip(np.add(DEAP_5_Standard_mean, DEAP_5_Standard_std), a_min=0, a_max=100)
DEAP_5_Standard_lowerlimit = np.clip(np.subtract(DEAP_5_Standard_mean, DEAP_5_Standard_std), a_min=0, a_max=100)

NEAT_5_Standard_mean = np.mean(NEAT_5_Standard, axis=0)
NEAT_5_Standard_std = np.std(NEAT_5_Standard, axis=0)
NEAT_5_Standard_upperlimit = np.clip(np.add(NEAT_5_Standard_mean, NEAT_5_Standard_std), a_min=0, a_max=100)
NEAT_5_Standard_lowerlimit = np.clip(np.subtract(NEAT_5_Standard_mean, NEAT_5_Standard_std), a_min=0, a_max=100)

DEAP_6_Standard_mean = np.mean(DEAP_6_Standard, axis=0)
DEAP_6_Standard_std = np.std(DEAP_6_Standard, axis=0)
DEAP_6_Standard_upperlimit = np.clip(np.add(DEAP_6_Standard_mean, DEAP_6_Standard_std), a_min=0, a_max=100)
DEAP_6_Standard_lowerlimit = np.clip(np.subtract(DEAP_6_Standard_mean, DEAP_6_Standard_std), a_min=0, a_max=100)

NEAT_6_Standard_mean = np.mean(NEAT_6_Standard, axis=0)
NEAT_6_Standard_std = np.std(NEAT_6_Standard, axis=0)
NEAT_6_Standard_upperlimit = np.clip(np.add(NEAT_6_Standard_mean, NEAT_6_Standard_std), a_min=0, a_max=100)
NEAT_6_Standard_lowerlimit = np.clip(np.subtract(NEAT_6_Standard_mean, NEAT_6_Standard_std), a_min=0, a_max=100)

DEAP_15_Standard_mean = np.mean(DEAP_15_Standard, axis=0)
DEAP_15_Standard_std = np.std(DEAP_15_Standard, axis=0)
DEAP_15_Standard_upperlimit = np.clip(np.add(DEAP_15_Standard_mean, DEAP_15_Standard_std), a_min=0, a_max=100)
DEAP_15_Standard_lowerlimit = np.clip(np.subtract(DEAP_15_Standard_mean, DEAP_15_Standard_std), a_min=0, a_max=100)

NEAT_15_Standard_mean = np.mean(NEAT_15_Standard, axis=0)
NEAT_15_Standard_std = np.std(NEAT_15_Standard, axis=0)
NEAT_15_Standard_upperlimit = np.clip(np.add(NEAT_15_Standard_mean, NEAT_15_Standard_std), a_min=0, a_max=100)
NEAT_15_Standard_lowerlimit = np.clip(np.subtract(NEAT_15_Standard_mean, NEAT_15_Standard_std), a_min=0, a_max=100)

DEAP_21_Standard_mean = np.mean(DEAP_21_Standard, axis=0)
DEAP_21_Standard_std = np.std(DEAP_21_Standard, axis=0)
DEAP_21_Standard_upperlimit = np.clip(np.add(DEAP_21_Standard_mean, DEAP_21_Standard_std), a_min=0, a_max=100)
DEAP_21_Standard_lowerlimit = np.clip(np.subtract(DEAP_21_Standard_mean, DEAP_21_Standard_std), a_min=0, a_max=100)

NEAT_21_Standard_mean = np.mean(NEAT_21_Standard, axis=0)
NEAT_21_Standard_std = np.std(NEAT_21_Standard, axis=0)
NEAT_21_Standard_upperlimit = np.clip(np.add(NEAT_21_Standard_mean, NEAT_21_Standard_std), a_min=0, a_max=100)
NEAT_21_Standard_lowerlimit = np.clip(np.subtract(NEAT_21_Standard_mean, NEAT_21_Standard_std), a_min=0, a_max=100)

DEAP_4_Logical_mean = np.mean(DEAP_4_Logical, axis=0)
DEAP_4_Logical_std = np.std(DEAP_4_Logical, axis=0)
DEAP_4_Logical_upperlimit = np.clip(np.add(DEAP_4_Logical_mean, DEAP_4_Logical_std), a_min=0, a_max=100)
DEAP_4_Logical_lowerlimit = np.clip(np.subtract(DEAP_4_Logical_mean, DEAP_4_Logical_std), a_min=0, a_max=100)

DEAP_5_Logical_mean = np.mean(DEAP_5_Logical, axis=0)
DEAP_5_Logical_std = np.std(DEAP_5_Logical, axis=0)
DEAP_5_Logical_upperlimit = np.clip(np.add(DEAP_5_Logical_mean, DEAP_5_Logical_std), a_min=0, a_max=100)
DEAP_5_Logical_lowerlimit = np.clip(np.subtract(DEAP_5_Logical_mean, DEAP_5_Logical_std), a_min=0, a_max=100)

DEAP_6_Logical_mean = np.mean(DEAP_6_Logical, axis=0)
DEAP_6_Logical_std = np.std(DEAP_6_Logical, axis=0)
DEAP_6_Logical_upperlimit = np.clip(np.add(DEAP_6_Logical_mean, DEAP_6_Logical_std), a_min=0, a_max=100)
DEAP_6_Logical_lowerlimit = np.clip(np.subtract(DEAP_6_Logical_mean, DEAP_6_Logical_std), a_min=0, a_max=100)

DEAP_15_Logical_mean = np.mean(DEAP_15_Logical, axis=0)
DEAP_15_Logical_std = np.std(DEAP_15_Logical, axis=0)
DEAP_15_Logical_upperlimit = np.clip(np.add(DEAP_15_Logical_mean, DEAP_15_Logical_std), a_min=0, a_max=100)
DEAP_15_Logical_lowerlimit = np.clip(np.subtract(DEAP_15_Logical_mean, DEAP_15_Logical_std), a_min=0, a_max=100)

DEAP_21_Logical_mean = np.mean(DEAP_21_Logical, axis=0)
DEAP_21_Logical_std = np.std(DEAP_21_Logical, axis=0)
DEAP_21_Logical_upperlimit = np.clip(np.add(DEAP_21_Logical_mean, DEAP_21_Logical_std), a_min=0, a_max=100)
DEAP_21_Logical_lowerlimit = np.clip(np.subtract(DEAP_21_Logical_mean, DEAP_21_Logical_std), a_min=0, a_max=100)

DEAP_4_Multiplication_mean = np.mean(DEAP_4_Multiplication, axis=0)
DEAP_4_Multiplication_std = np.std(DEAP_4_Multiplication, axis=0)
DEAP_4_Multiplication_upperlimit = np.clip(np.add(DEAP_4_Multiplication_mean, DEAP_4_Multiplication_std), a_min=0, a_max=100)
DEAP_4_Multiplication_lowerlimit = np.clip(np.subtract(DEAP_4_Multiplication_mean, DEAP_4_Multiplication_std), a_min=0, a_max=100)

DEAP_5_Multiplication_mean = np.mean(DEAP_5_Multiplication, axis=0)
DEAP_5_Multiplication_std = np.std(DEAP_5_Multiplication, axis=0)
DEAP_5_Multiplication_upperlimit = np.clip(np.add(DEAP_5_Multiplication_mean, DEAP_5_Multiplication_std), a_min=0, a_max=100)
DEAP_5_Multiplication_lowerlimit = np.clip(np.subtract(DEAP_5_Multiplication_mean, DEAP_5_Multiplication_std), a_min=0, a_max=100)

DEAP_4_Modified_mean = np.mean(DEAP_4_Modified, axis=0)
DEAP_4_Modified_std = np.std(DEAP_4_Modified, axis=0)
DEAP_4_Modified_upperlimit = np.clip(np.add(DEAP_4_Modified_mean, DEAP_4_Modified_std), a_min=0, a_max=100)
DEAP_4_Modified_lowerlimit = np.clip(np.subtract(DEAP_4_Modified_mean, DEAP_4_Modified_std), a_min=0, a_max=100)

DEAP_5_Modified_mean = np.mean(DEAP_5_Modified, axis=0)
DEAP_5_Modified_std = np.std(DEAP_5_Modified, axis=0)
DEAP_5_Modified_upperlimit = np.clip(np.add(DEAP_5_Modified_mean, DEAP_5_Modified_std), a_min=0, a_max=100)
DEAP_5_Modified_lowerlimit = np.clip(np.subtract(DEAP_5_Modified_mean, DEAP_5_Modified_std), a_min=0, a_max=100)

DEAP_6_Modified_mean = np.mean(DEAP_6_Modified, axis=0)
DEAP_6_Modified_std = np.std(DEAP_6_Modified, axis=0)
DEAP_6_Modified_upperlimit = np.clip(np.add(DEAP_6_Modified_mean, DEAP_6_Modified_std), a_min=0, a_max=100)
DEAP_6_Modified_lowerlimit = np.clip(np.subtract(DEAP_6_Modified_mean, DEAP_6_Modified_std), a_min=0, a_max=100)

DEAP_15_Modified_mean = np.mean(DEAP_15_Modified, axis=0)
DEAP_15_Modified_std = np.std(DEAP_15_Modified, axis=0)
DEAP_15_Modified_upperlimit = np.clip(np.add(DEAP_15_Modified_mean, DEAP_15_Modified_std), a_min=0, a_max=100)
DEAP_15_Modified_lowerlimit = np.clip(np.subtract(DEAP_15_Modified_mean, DEAP_15_Modified_std), a_min=0, a_max=100)

DEAP_21_Modified_mean = np.mean(DEAP_21_Modified, axis=0)
DEAP_21_Modified_std = np.std(DEAP_21_Modified, axis=0)
DEAP_21_Modified_upperlimit = np.clip(np.add(DEAP_21_Modified_mean, DEAP_21_Modified_std), a_min=0, a_max=100)
DEAP_21_Modified_lowerlimit = np.clip(np.subtract(DEAP_21_Modified_mean, DEAP_21_Modified_std), a_min=0, a_max=100)


'''
    Plot Results
'''

# NEAT vs DEAP
plt.figure(1)
plt.plot(axis_x_250, DEAP_4_Standard_mean, linewidth=1, label="4-depth GP")
plt.fill_between(axis_x_250, DEAP_4_Standard_lowerlimit, DEAP_4_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_500, NEAT_4_Standard_mean, linewidth=1, label="4-depth NEAT")
plt.fill_between(axis_x_500, NEAT_4_Standard_lowerlimit, NEAT_4_Standard_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/DEAP_vs_NEAT_4.png", bbox_inches='tight')

plt.figure(2)
plt.plot(axis_x_250, DEAP_5_Standard_mean, linewidth=1, label="5-depth GP")
plt.fill_between(axis_x_250, DEAP_5_Standard_lowerlimit, DEAP_5_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_500, NEAT_5_Standard_mean, linewidth=1, label="5-depth NEAT")
plt.fill_between(axis_x_500, NEAT_5_Standard_lowerlimit, NEAT_5_Standard_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/DEAP_vs_NEAT_5.png", bbox_inches='tight')

plt.figure(3)
plt.plot(axis_x_250, DEAP_6_Standard_mean, linewidth=1, label="6-depth GP")
plt.fill_between(axis_x_250, DEAP_6_Standard_lowerlimit, DEAP_6_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_500, NEAT_6_Standard_mean, linewidth=1, label="6-depth NEAT")
plt.fill_between(axis_x_500, NEAT_6_Standard_lowerlimit, NEAT_6_Standard_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/DEAP_vs_NEAT_6.png", bbox_inches='tight')

plt.figure(4)
plt.plot(axis_x_250, DEAP_15_Standard_mean, linewidth=1, label="15-depth GP")
plt.fill_between(axis_x_250, DEAP_15_Standard_lowerlimit, DEAP_15_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_500, NEAT_15_Standard_mean, linewidth=1, label="15-depth NEAT")
plt.fill_between(axis_x_500, NEAT_15_Standard_lowerlimit, NEAT_15_Standard_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/DEAP_vs_NEAT_15.png", bbox_inches='tight')

plt.figure(5)
plt.plot(axis_x_250, DEAP_21_Standard_mean, linewidth=1, label="21-depth GP")
plt.fill_between(axis_x_250, DEAP_21_Standard_lowerlimit, DEAP_21_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_500, NEAT_21_Standard_mean, linewidth=1, label="21-depth NEAT")
plt.fill_between(axis_x_500, NEAT_21_Standard_lowerlimit, NEAT_21_Standard_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/DEAP_vs_NEAT_21.png", bbox_inches='tight')

plt.legend(loc=legend_loc)
plt.show()

# Div vs Full
plt.figure(6)
plt.plot(axis_x_250, DEAP_4_Standard_mean, linewidth=1, label="4-depth Div")
plt.fill_between(axis_x_250, DEAP_4_Standard_lowerlimit, DEAP_4_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_4_Logical_mean, linewidth=1, label="4-depth Full")
plt.fill_between(axis_x_250, DEAP_4_Logical_lowerlimit, DEAP_4_Logical_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Logical_4.png", bbox_inches='tight')

plt.figure(7)
plt.plot(axis_x_250, DEAP_5_Standard_mean, linewidth=1, label="5-depth Div")
plt.fill_between(axis_x_250, DEAP_5_Standard_lowerlimit, DEAP_5_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_5_Logical_mean, linewidth=1, label="5-depth Full")
plt.fill_between(axis_x_250, DEAP_5_Logical_lowerlimit, DEAP_5_Logical_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Logical_5.png", bbox_inches='tight')

plt.figure(8)
plt.plot(axis_x_250, DEAP_6_Standard_mean, linewidth=1, label="6-depth Div")
plt.fill_between(axis_x_250, DEAP_6_Standard_lowerlimit, DEAP_6_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_6_Logical_mean, linewidth=1, label="6-depth Full")
plt.fill_between(axis_x_250, DEAP_6_Logical_lowerlimit, DEAP_6_Logical_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Logical_6.png", bbox_inches='tight')

plt.figure(9)
plt.plot(axis_x_250, DEAP_15_Standard_mean, linewidth=1, label="15-depth Div")
plt.fill_between(axis_x_250, DEAP_15_Standard_lowerlimit, DEAP_15_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_15_Logical_mean, linewidth=1, label="15-depth Full")
plt.fill_between(axis_x_250, DEAP_15_Logical_lowerlimit, DEAP_15_Logical_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Logical_15.png", bbox_inches='tight')

plt.figure(10)
plt.plot(axis_x_250, DEAP_21_Standard_mean, linewidth=1, label="21-depth Div")
plt.fill_between(axis_x_250, DEAP_21_Standard_lowerlimit, DEAP_21_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_21_Logical_mean, linewidth=1, label="21-depth Full")
plt.fill_between(axis_x_250, DEAP_21_Logical_lowerlimit, DEAP_21_Logical_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Logical_21.png", bbox_inches='tight')

plt.legend(loc=legend_loc)
plt.show()

# Div vs Multiplication
plt.figure(11)
plt.plot(axis_x_250, DEAP_4_Standard_mean, linewidth=1, label="4-depth Div")
plt.fill_between(axis_x_250, DEAP_4_Standard_lowerlimit, DEAP_4_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_4_Multiplication_mean, linewidth=1, label="4-depth Multiplication")
plt.fill_between(axis_x_250, DEAP_4_Multiplication_lowerlimit, DEAP_4_Multiplication_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Multiplication_4.png", bbox_inches='tight')

plt.figure(12)
plt.plot(axis_x_250, DEAP_5_Standard_mean, linewidth=1, label="5-depth Div")
plt.fill_between(axis_x_250, DEAP_5_Standard_lowerlimit, DEAP_5_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_5_Multiplication_mean, linewidth=1, label="5-depth Multiplication")
plt.fill_between(axis_x_250, DEAP_5_Multiplication_lowerlimit, DEAP_5_Multiplication_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Multiplication_5.png", bbox_inches='tight')

plt.legend(loc=legend_loc)
plt.show()

# Original vs Modified
plt.figure(13)
plt.plot(axis_x_250, DEAP_4_Standard_mean, linewidth=1, label="4-depth Original")
plt.fill_between(axis_x_250, DEAP_4_Standard_lowerlimit, DEAP_4_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_4_Modified_mean, linewidth=1, label="4-depth Modified")
plt.fill_between(axis_x_250, DEAP_4_Modified_lowerlimit, DEAP_4_Modified_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Modified_4.png", bbox_inches='tight')

plt.figure(14)
plt.plot(axis_x_250, DEAP_5_Standard_mean, linewidth=1, label="5-depth Original")
plt.fill_between(axis_x_250, DEAP_5_Standard_lowerlimit, DEAP_5_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_5_Modified_mean, linewidth=1, label="5-depth Modified")
plt.fill_between(axis_x_250, DEAP_5_Modified_lowerlimit, DEAP_5_Modified_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Modified_5.png", bbox_inches='tight')

plt.figure(15)
plt.plot(axis_x_250, DEAP_6_Standard_mean, linewidth=1, label="6-depth Original")
plt.fill_between(axis_x_250, DEAP_6_Standard_lowerlimit, DEAP_6_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_6_Modified_mean, linewidth=1, label="6-depth Modified")
plt.fill_between(axis_x_250, DEAP_6_Modified_lowerlimit, DEAP_6_Modified_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Modified_6.png", bbox_inches='tight')

plt.figure(16)
plt.plot(axis_x_250, DEAP_15_Standard_mean, linewidth=1, label="15-depth Original")
plt.fill_between(axis_x_250, DEAP_15_Standard_lowerlimit, DEAP_15_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_15_Modified_mean, linewidth=1, label="15-depth Modified")
plt.fill_between(axis_x_250, DEAP_15_Modified_lowerlimit, DEAP_15_Modified_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Modified_15.png", bbox_inches='tight')

plt.figure(17)
plt.plot(axis_x_250, DEAP_21_Standard_mean, linewidth=1, label="21-depth Original")
plt.fill_between(axis_x_250, DEAP_21_Standard_lowerlimit, DEAP_21_Standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, DEAP_21_Modified_mean, linewidth=1, label="21-depth Modified")
plt.fill_between(axis_x_250, DEAP_21_Modified_lowerlimit, DEAP_21_Modified_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Recall/Standard_vs_Modified_21.png", bbox_inches='tight')

plt.legend(loc=legend_loc)
plt.show()