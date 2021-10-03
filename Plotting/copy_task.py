import matplotlib.pyplot as plt
import pickle
import os
import csv

local_dir = os.path.dirname(__file__)
axis_x = list(range(0, 251))
y_label = "Success Percentage"
x_label = "Training Generations"

# 8-bit Report
plt.figure(1)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/8-bit-report/8-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Copy Task/DEAP_Standard.png", bbox_inches='tight')

# 8-bit Report Logical
plt.figure(2)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/8-bit-report/8-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Copy Task/DEAP_Logical.png", bbox_inches='tight')

# 8-bit Report Modified
plt.figure(3)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/8-bit-report/8-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Copy Task/DEAP_Modified.png", bbox_inches='tight')

# 8-bit Report Multiplication
plt.figure(4)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Copy Task/8-bit/8-bit-report/8-progress_report_mul' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Copy Task/DEAP_Multiplication.png", bbox_inches='tight')

# NEAT 8-bit Report
plt.figure(5)
axis_x = list(range(0, 500))
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/8-bit CopyTask/8-bit-report/fitness_history' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            info.append(float(*row))
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Copy Task/NEAT_Standard.png", bbox_inches='tight')
plt.show()