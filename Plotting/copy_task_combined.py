import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import csv

local_dir = os.path.dirname(__file__)
axis_x_250 = list(range(0, 251))
axis_x_500 = list(range(0, 500))
y_label = "Success Percentage"
x_label = "Training Generations"
legend_loc = "lower right"

'''
    Load Data
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

# DEAP Logical Operators
logic_info = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/8-bit-report/8-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    logic_info.append(info)

# DEAP Modified Task
mod_info = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/8-bit-report/8-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    mod_info.append(info)

# DEAP Multiplication Operator
mul_info = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/8-bit-report/8-progress_report_mul' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    mul_info.append(info)


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

logic_mean = np.mean(logic_info, axis=0)
logic_std = np.std(logic_info, axis=0)
DEAP_logic_upperlimit = np.clip(np.add(logic_mean, logic_std), a_min=0, a_max=100)
DEAP_logic_lowerlimit = np.clip(np.subtract(logic_mean, logic_std), a_min=0, a_max=100)

mod_mean = np.mean(mod_info, axis=0)
mod_std = np.std(mod_info, axis=0)
DEAP_mod_upperlimit = np.clip(np.add(mod_mean, mod_std), a_min=0, a_max=100)
DEAP_mod_lowerlimit = np.clip(np.subtract(mod_mean, mod_std), a_min=0, a_max=100)

mul_mean = np.mean(mul_info, axis=0)
mul_std = np.std(mul_info, axis=0)
DEAP_mul_upperlimit = np.clip(np.add(mul_mean, mul_std), a_min=0, a_max=100)
DEAP_mul_lowerlimit = np.clip(np.subtract(mul_mean, mul_std), a_min=0, a_max=100)


'''
    Plot Results
'''

# DEAP vs NEAT
plt.figure(1)
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="DEAP")
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, alpha=.3)
plt.plot(axis_x_500, NEAT_standard_mean, linewidth=1, label="NEAT")
plt.fill_between(axis_x_500, NEAT_standard_lowerlimit, NEAT_standard_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/DEAP_vs_NEAT.png", bbox_inches='tight')

# Standard vs Logical
plt.figure(2)
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="Standard")
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, logic_mean, linewidth=1, label="Logical")
plt.fill_between(axis_x_250, DEAP_logic_lowerlimit, DEAP_logic_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/Standard_vs_logical.png", bbox_inches='tight')

# Standard vs Modified
plt.figure(3)
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="Standard")
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, mod_mean, linewidth=1, label="Modified")
plt.fill_between(axis_x_250, DEAP_mod_lowerlimit, DEAP_mod_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/Standard_vs_Modified.png", bbox_inches='tight')

# Standard vs Multiplication
plt.figure(4)
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="Standard")
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, mul_mean, linewidth=1, label="Multiplication")
plt.fill_between(axis_x_250, DEAP_mul_lowerlimit, DEAP_mul_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/Standard_vs_Multiplication.png", bbox_inches='tight')

# Combined figure
plt.figure(5)

#  Plot standard
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="Standard")
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, alpha=.3)

# Plot Logical
plt.plot(axis_x_250, logic_mean, linewidth=1, label="Logical")
plt.fill_between(axis_x_250, DEAP_logic_lowerlimit, DEAP_logic_upperlimit, alpha=.3)

#  Plot modified
plt.plot(axis_x_250, mod_mean, linewidth=1, label="Modified")
plt.fill_between(axis_x_250, DEAP_mod_lowerlimit, DEAP_mod_upperlimit, alpha=.3)

#  Plot multiplication
plt.plot(axis_x_250, mul_mean, linewidth=1, label="Multiplication")
plt.fill_between(axis_x_250, DEAP_mul_lowerlimit, DEAP_mul_upperlimit, alpha=.3)

plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/DEAP_combined.png", bbox_inches='tight')
plt.show()