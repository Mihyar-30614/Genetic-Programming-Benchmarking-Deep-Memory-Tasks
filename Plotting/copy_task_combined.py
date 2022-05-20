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
# 8-bit Report DEAP
path = os.path.join(local_dir, '../DEAP/Copy Task/reports/8_report_std')
with open(path, 'rb') as f:
    DEAP_Standard = list(pickle.load(f).values())

# 8-bit Report NEAT
NEAT_Standard = []
path = os.path.join(local_dir, '../NEAT/Copy Task/reports/8_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    NEAT_Standard.append(info)

# DEAP Logical Operators
path = os.path.join(local_dir, '../DEAP/Copy Task/reports/8_report_log')
with open(path, 'rb') as f:
    logic_info = list(pickle.load(f).values())

# DEAP Modified Task
path = os.path.join(local_dir, '../DEAP/Copy Task/reports/8_report_mod')
with open(path, 'rb') as f:
    mod_info = list(pickle.load(f).values())

# DEAP Multiplication Operator
path = os.path.join(local_dir, '../DEAP/Copy Task/reports/8_report_mul')
with open(path, 'rb') as f:
    mul_info = list(pickle.load(f).values())

# 8-bit-vector
DEAP_Vector = []
path = os.path.join(local_dir, '../DEAP/Copy Task/reports/8_report_vec')
with open(path, 'rb') as f:
    DEAP_Vector = list(pickle.load(f).values())


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

DEAP_vector_mean = np.mean(DEAP_Vector, axis=0)
DEAP_vector_std = np.std(DEAP_Vector, axis=0)
DEAP_vector_upperlimit = np.clip(np.add(DEAP_vector_mean, DEAP_vector_std), a_min=0, a_max=100)
DEAP_vector_lowerlimit = np.clip(np.subtract(DEAP_vector_mean, DEAP_vector_std), a_min=0, a_max=100)


'''
    Plot Results
'''

# DEAP vs NEAT
plt.figure(1)
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="GP")
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, alpha=.3)
plt.plot(axis_x_500, NEAT_standard_mean, linewidth=1, label="NEAT")
plt.fill_between(axis_x_500, NEAT_standard_lowerlimit, NEAT_standard_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/DEAP_vs_NEAT.png", bbox_inches='tight')

# Div vs Full
plt.figure(2)
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="Div")
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, logic_mean, linewidth=1, label="Full")
plt.fill_between(axis_x_250, DEAP_logic_lowerlimit, DEAP_logic_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/Standard_vs_logical.png", bbox_inches='tight')

# Original vs Modified
plt.figure(3)
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="Original")
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, mod_mean, linewidth=1, label="Modified")
plt.fill_between(axis_x_250, DEAP_mod_lowerlimit, DEAP_mod_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/Standard_vs_Modified.png", bbox_inches='tight')

# Standard vs Multiplication
plt.figure(4)
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="Div")
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, alpha=.3)
plt.plot(axis_x_250, mul_mean, linewidth=1, label="Multiplication")
plt.fill_between(axis_x_250, DEAP_mul_lowerlimit, DEAP_mul_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/Standard_vs_Multiplication.png", bbox_inches='tight')

# Combined figure
plt.figure(5)

#  Plot Div
plt.plot(axis_x_250, DEAP_standard_mean, linewidth=1, label="Div")
plt.fill_between(axis_x_250, DEAP_standard_lowerlimit, DEAP_standard_upperlimit, alpha=.3)

# Plot Logical
plt.plot(axis_x_250, logic_mean, linewidth=1, label="Full")
plt.fill_between(axis_x_250, DEAP_logic_lowerlimit, DEAP_logic_upperlimit, alpha=.3)

#  Plot modified
plt.plot(axis_x_250, mod_mean, linewidth=1, label="Modified")
plt.fill_between(axis_x_250, DEAP_mod_lowerlimit, DEAP_mod_upperlimit, alpha=.3)

#  Plot multiplication
plt.plot(axis_x_250, mul_mean, linewidth=1, label="Multiplication")
plt.fill_between(axis_x_250, DEAP_mul_lowerlimit, DEAP_mul_upperlimit, alpha=.3)

plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/DEAP_combined.png", bbox_inches='tight')

# 8-bit-vector
plt.figure(6)
plt.plot(axis_x_250, DEAP_vector_mean, linewidth=1, label="8-bit-vector")
plt.fill_between(axis_x_250, DEAP_vector_lowerlimit, DEAP_vector_upperlimit, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Copy Task/8-bit-vector.png", bbox_inches='tight')

plt.show()