import matplotlib.pyplot as plt
import pickle
import os
import csv

local_dir = os.path.dirname(__file__)
axis_x_250 = list(range(251))
axis_x_500 = list(range(500))
y_label = "Success Percentage"
x_label = "Training Generations"

# 8-bit Report
plt.figure(1)
path = os.path.join(local_dir, '../DEAP/Copy Task/reports/8_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Copy Task/DEAP_Standard.png", bbox_inches='tight')

# 8-bit Report Logical
plt.figure(2)
path = os.path.join(local_dir, '../DEAP/Copy Task/reports/8_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Copy Task/DEAP_Logical.png", bbox_inches='tight')

# 8-bit Report Modified
plt.figure(3)
path = os.path.join(local_dir, '../DEAP/Copy Task/reports/8_report_mod')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Copy Task/DEAP_Modified.png", bbox_inches='tight')

# 8-bit Report Multiplication
plt.figure(4)
path = os.path.join(local_dir, '../DEAP/Copy Task/reports/8_report_mul')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Copy Task/DEAP_Multiplication.png", bbox_inches='tight')

# NEAT 8-bit Report
plt.figure(5)
path = os.path.join(local_dir, '../NEAT/Copy Task/reports/8_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Copy Task/NEAT_Standard.png", bbox_inches='tight')

# 8-bit-vector Report
plt.figure(6)
path = os.path.join(local_dir, '../DEAP/Copy Task/reports/8_report_vec')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Copy Task/DEAP_Vector_Standard.png", bbox_inches='tight')

plt.show()