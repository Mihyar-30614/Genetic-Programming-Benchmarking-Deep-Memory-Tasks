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
# 4-deep Report DEAP
DEAP_4_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/4-deep-report/4-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_4_Standard.append(info)

# 4-deep Report NEAT
NEAT_4_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Classification/4-deep-report/4-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
    NEAT_4_Standard.append(info)

# 5-deep Report DEAP
DEAP_5_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/5-deep-report/5-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_5_Standard.append(info)

# 5-deep Report NEAT
NEAT_5_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Classification/5-deep-report/5-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
    NEAT_5_Standard.append(info)

# 6-deep Report DEAP
DEAP_6_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/6-deep-report/6-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_6_Standard.append(info)

# 6-deep Report NEAT
NEAT_6_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Classification/6-deep-report/6-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
    NEAT_6_Standard.append(info)

# 15-deep Report DEAP
DEAP_15_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/15-deep-report/15-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_15_Standard.append(info)

# 15-deep Report NEAT
NEAT_15_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Classification/15-deep-report/15-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
    NEAT_15_Standard.append(info)

# 21-deep Report DEAP
DEAP_21_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/21-deep-report/21-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_21_Standard.append(info)

# 21-deep Report NEAT
NEAT_21_Standard = []
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Classification/21-deep-report/21-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
    NEAT_21_Standard.append(info)

# 4-deep Report DEAP Logical
DEAP_4_Logical = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/4-deep-report/4-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_4_Logical.append(info)

# 5-deep Report DEAP Logical
DEAP_5_Logical = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/5-deep-report/5-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_5_Logical.append(info)

# 6-deep Report DEAP Logical
DEAP_6_Logical = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/6-deep-report/6-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_6_Logical.append(info)

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

# 4-deep Report DEAP Multiplication
DEAP_4_Multiplication = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/4-deep-report/4-progress_report_mul' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_4_Multiplication.append(info)

# 5-deep Report DEAP Multiplication
DEAP_5_Multiplication = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/5-deep-report/5-progress_report_mul' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_5_Multiplication.append(info)

# 4-deep Report DEAP Modified 0.5
DEAP_4_Modified_5 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/4-deep-report/4-progress_report_0.5_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_4_Modified_5.append(info)

# 5-deep Report DEAP Modified 0.5
DEAP_5_Modified_5 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/5-deep-report/5-progress_report_0.5_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_5_Modified_5.append(info)

# 6-deep Report DEAP Modified 0.5
DEAP_6_Modified_5 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/6-deep-report/6-progress_report_0.5_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_6_Modified_5.append(info)

# 15-deep Report DEAP Modified 0.5
DEAP_15_Modified_5 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/15-deep-report/15-progress_report_0.5_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_15_Modified_5.append(info)

# 21-deep Report DEAP Modified 0.5
DEAP_21_Modified_5 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/21-deep-report/21-progress_report_0.5_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_21_Modified_5.append(info)

# 4-deep Report DEAP Modified 0.25
DEAP_4_Modified_25 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/4-deep-report/4-progress_report_0.25_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_4_Modified_25.append(info)

# 5-deep Report DEAP Modified 0.25
DEAP_5_Modified_25 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/5-deep-report/5-progress_report_0.25_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_5_Modified_25.append(info)

# 6-deep Report DEAP Modified 0.25
DEAP_6_Modified_25 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/6-deep-report/6-progress_report_0.25_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_6_Modified_25.append(info)

# 15-deep Report DEAP Modified 0.25
DEAP_15_Modified_25 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/15-deep-report/15-progress_report_0.25_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_15_Modified_25.append(info)

# 21-deep Report DEAP Modified 0.25
DEAP_21_Modified_25 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/21-deep-report/21-progress_report_0.25_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_21_Modified_25.append(info)

# 4-deep Report DEAP Modified 0.125
DEAP_4_Modified_125 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/4-deep-report/4-progress_report_0.125_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_4_Modified_125.append(info)

# 5-deep Report DEAP Modified 0.125
DEAP_5_Modified_125 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/5-deep-report/5-progress_report_0.125_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_5_Modified_125.append(info)

# 6-deep Report DEAP Modified 0.125
DEAP_6_Modified_125 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/6-deep-report/6-progress_report_0.125_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_6_Modified_125.append(info)

# 15-deep Report DEAP Modified 0.125
DEAP_15_Modified_125 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/15-deep-report/15-progress_report_0.125_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_15_Modified_125.append(info)

# 21-deep Report DEAP Modified 0.125
DEAP_21_Modified_125 = []
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Classification/21-deep-report/21-progress_report_0.125_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    DEAP_21_Modified_125.append(info)

'''
    Calculate Mean and Standard Deviation
'''

DEAP_4_Standard_mean = np.mean(DEAP_4_Standard, axis=0)
DEAP_4_Standard_std = np.std(DEAP_4_Standard, axis=0)

NEAT_4_Standard_mean = np.mean(NEAT_4_Standard, axis=0)
NEAT_4_Standard_std = np.std(NEAT_4_Standard, axis=0)

DEAP_5_Standard_mean = np.mean(DEAP_5_Standard, axis=0)
DEAP_5_Standard_std = np.std(DEAP_5_Standard, axis=0)

NEAT_5_Standard_mean = np.mean(NEAT_5_Standard, axis=0)
NEAT_5_Standard_std = np.std(NEAT_5_Standard, axis=0)

DEAP_6_Standard_mean = np.mean(DEAP_6_Standard, axis=0)
DEAP_6_Standard_std = np.std(DEAP_6_Standard, axis=0)

NEAT_6_Standard_mean = np.mean(NEAT_6_Standard, axis=0)
NEAT_6_Standard_std = np.std(NEAT_6_Standard, axis=0)

DEAP_15_Standard_mean = np.mean(DEAP_15_Standard, axis=0)
DEAP_15_Standard_std = np.std(DEAP_15_Standard, axis=0)

NEAT_15_Standard_mean = np.mean(NEAT_15_Standard, axis=0)
NEAT_15_Standard_std = np.std(NEAT_15_Standard, axis=0)

DEAP_21_Standard_mean = np.mean(DEAP_21_Standard, axis=0)
DEAP_21_Standard_std = np.std(DEAP_21_Standard, axis=0)

NEAT_21_Standard_mean = np.mean(NEAT_21_Standard, axis=0)
NEAT_21_Standard_std = np.std(NEAT_21_Standard, axis=0)

DEAP_4_Logical_mean = np.mean(DEAP_4_Logical, axis=0)
DEAP_4_Logical_std = np.std(DEAP_4_Logical, axis=0)

DEAP_5_Logical_mean = np.mean(DEAP_5_Logical, axis=0)
DEAP_5_Logical_std = np.std(DEAP_5_Logical, axis=0)

DEAP_6_Logical_mean = np.mean(DEAP_6_Logical, axis=0)
DEAP_6_Logical_std = np.std(DEAP_6_Logical, axis=0)

DEAP_15_Logical_mean = np.mean(DEAP_15_Logical, axis=0)
DEAP_15_Logical_std = np.std(DEAP_15_Logical, axis=0)

DEAP_21_Logical_mean = np.mean(DEAP_21_Logical, axis=0)
DEAP_21_Logical_std = np.std(DEAP_21_Logical, axis=0)

DEAP_4_Multiplication_mean = np.mean(DEAP_4_Multiplication, axis=0)
DEAP_4_Multiplication_std = np.std(DEAP_4_Multiplication, axis=0)

DEAP_5_Multiplication_mean = np.mean(DEAP_5_Multiplication, axis=0)
DEAP_5_Multiplication_std = np.std(DEAP_5_Multiplication, axis=0)

DEAP_4_Modified_5_mean = np.mean(DEAP_4_Modified_5, axis=0)
DEAP_4_Modified_5_std = np.std(DEAP_4_Modified_5, axis=0)

DEAP_5_Modified_5_mean = np.mean(DEAP_5_Modified_5, axis=0)
DEAP_5_Modified_5_std = np.std(DEAP_5_Modified_5, axis=0)

DEAP_6_Modified_5_mean = np.mean(DEAP_6_Modified_5, axis=0)
DEAP_6_Modified_5_std = np.std(DEAP_6_Modified_5, axis=0)

DEAP_15_Modified_5_mean = np.mean(DEAP_15_Modified_5, axis=0)
DEAP_15_Modified_5_std = np.std(DEAP_15_Modified_5, axis=0)

DEAP_21_Modified_5_mean = np.mean(DEAP_21_Modified_5, axis=0)
DEAP_21_Modified_5_std = np.std(DEAP_21_Modified_5, axis=0)

DEAP_4_Modified_25_mean = np.mean(DEAP_4_Modified_25, axis=0)
DEAP_4_Modified_25_std = np.std(DEAP_4_Modified_25, axis=0)

DEAP_5_Modified_25_mean = np.mean(DEAP_5_Modified_25, axis=0)
DEAP_5_Modified_25_std = np.std(DEAP_5_Modified_25, axis=0)

DEAP_6_Modified_25_mean = np.mean(DEAP_6_Modified_25, axis=0)
DEAP_6_Modified_25_std = np.std(DEAP_6_Modified_25, axis=0)

DEAP_15_Modified_25_mean = np.mean(DEAP_15_Modified_25, axis=0)
DEAP_15_Modified_25_std = np.std(DEAP_15_Modified_25, axis=0)

DEAP_21_Modified_25_mean = np.mean(DEAP_21_Modified_25, axis=0)
DEAP_21_Modified_25_std = np.std(DEAP_21_Modified_25, axis=0)

DEAP_4_Modified_125_mean = np.mean(DEAP_4_Modified_125, axis=0)
DEAP_4_Modified_125_std = np.std(DEAP_4_Modified_125, axis=0)

DEAP_5_Modified_125_mean = np.mean(DEAP_5_Modified_125, axis=0)
DEAP_5_Modified_125_std = np.std(DEAP_5_Modified_125, axis=0)

DEAP_6_Modified_125_mean = np.mean(DEAP_6_Modified_125, axis=0)
DEAP_6_Modified_125_std = np.std(DEAP_6_Modified_125, axis=0)

DEAP_15_Modified_125_mean = np.mean(DEAP_15_Modified_125, axis=0)
DEAP_15_Modified_125_std = np.std(DEAP_15_Modified_125, axis=0)

DEAP_21_Modified_125_mean = np.mean(DEAP_21_Modified_125, axis=0)
DEAP_21_Modified_125_std = np.std(DEAP_21_Modified_125, axis=0)


'''
    Plot Results
'''

# DEAP vs NEAT
plt.figure(1)
plt.plot(axis_x_250, DEAP_4_Standard_mean, linewidth=1, label="4-deep DEAP")
plt.fill_between(axis_x_250, DEAP_4_Standard_mean - DEAP_4_Standard_std, DEAP_4_Standard_mean + DEAP_4_Standard_std, alpha=.3)
plt.plot(axis_x_500, NEAT_4_Standard_mean, linewidth=1, label="4-deep NEAT")
plt.fill_between(axis_x_500, NEAT_4_Standard_mean - NEAT_4_Standard_std, NEAT_4_Standard_mean + NEAT_4_Standard_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/DEAP_vs_NEAT_4.png", bbox_inches='tight')

plt.figure(2)
plt.plot(axis_x_250, DEAP_5_Standard_mean, linewidth=1, label="5-deep DEAP")
plt.fill_between(axis_x_250, DEAP_5_Standard_mean - DEAP_5_Standard_std, DEAP_5_Standard_mean + DEAP_5_Standard_std, alpha=.3)
plt.plot(axis_x_500, NEAT_5_Standard_mean, linewidth=1, label="5-deep NEAT")
plt.fill_between(axis_x_500, NEAT_5_Standard_mean - NEAT_5_Standard_std, NEAT_5_Standard_mean + NEAT_5_Standard_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/DEAP_vs_NEAT_5.png", bbox_inches='tight')

plt.figure(3)
plt.plot(axis_x_250, DEAP_6_Standard_mean, linewidth=1, label="6-deep DEAP")
plt.fill_between(axis_x_250, DEAP_6_Standard_mean - DEAP_6_Standard_std, DEAP_6_Standard_mean + DEAP_6_Standard_std, alpha=.3)
plt.plot(axis_x_500, NEAT_6_Standard_mean, linewidth=1, label="6-deep NEAT")
plt.fill_between(axis_x_500, NEAT_6_Standard_mean - NEAT_6_Standard_std, NEAT_6_Standard_mean + NEAT_6_Standard_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/DEAP_vs_NEAT_6.png", bbox_inches='tight')

plt.figure(4)
plt.plot(axis_x_250, DEAP_15_Standard_mean, linewidth=1, label="15-deep DEAP")
plt.fill_between(axis_x_250, DEAP_15_Standard_mean - DEAP_15_Standard_std, DEAP_15_Standard_mean + DEAP_15_Standard_std, alpha=.3)
plt.plot(axis_x_500, NEAT_15_Standard_mean, linewidth=1, label="15-deep NEAT")
plt.fill_between(axis_x_500, NEAT_15_Standard_mean - NEAT_15_Standard_std, NEAT_15_Standard_mean + NEAT_15_Standard_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/DEAP_vs_NEAT_15.png", bbox_inches='tight')

plt.figure(5)
plt.plot(axis_x_250, DEAP_21_Standard_mean, linewidth=1, label="21-deep DEAP")
plt.fill_between(axis_x_250, DEAP_21_Standard_mean - DEAP_21_Standard_std, DEAP_21_Standard_mean + DEAP_21_Standard_std, alpha=.3)
plt.plot(axis_x_500, NEAT_21_Standard_mean, linewidth=1, label="21-deep NEAT")
plt.fill_between(axis_x_500, NEAT_21_Standard_mean - NEAT_21_Standard_std, NEAT_21_Standard_mean + NEAT_21_Standard_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/DEAP_vs_NEAT_21.png", bbox_inches='tight')

plt.legend(loc=legend_loc)
plt.show()

# Standard vs Logical
plt.figure(6)
plt.plot(axis_x_250, DEAP_4_Standard_mean, linewidth=1, label="4-deep Standard")
plt.fill_between(axis_x_250, DEAP_4_Standard_mean - DEAP_4_Standard_std, DEAP_4_Standard_mean + DEAP_4_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_4_Logical_mean, linewidth=1, label="4-deep Logical")
plt.fill_between(axis_x_250, DEAP_4_Logical_mean - DEAP_4_Logical_std, DEAP_4_Logical_mean + DEAP_4_Logical_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Logical_4.png", bbox_inches='tight')

plt.figure(7)
plt.plot(axis_x_250, DEAP_5_Standard_mean, linewidth=1, label="5-deep Standard")
plt.fill_between(axis_x_250, DEAP_5_Standard_mean - DEAP_5_Standard_std, DEAP_5_Standard_mean + DEAP_5_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_5_Logical_mean, linewidth=1, label="5-deep Logical")
plt.fill_between(axis_x_250, DEAP_5_Logical_mean - DEAP_5_Logical_std, DEAP_5_Logical_mean + DEAP_5_Logical_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Logical_5.png", bbox_inches='tight')

plt.figure(8)
plt.plot(axis_x_250, DEAP_6_Standard_mean, linewidth=1, label="6-deep Standard")
plt.fill_between(axis_x_250, DEAP_6_Standard_mean - DEAP_6_Standard_std, DEAP_6_Standard_mean + DEAP_6_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_6_Logical_mean, linewidth=1, label="6-deep Logical")
plt.fill_between(axis_x_250, DEAP_6_Logical_mean - DEAP_6_Logical_std, DEAP_6_Logical_mean + DEAP_6_Logical_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Logical_6.png", bbox_inches='tight')

plt.figure(9)
plt.plot(axis_x_250, DEAP_15_Standard_mean, linewidth=1, label="15-deep Standard")
plt.fill_between(axis_x_250, DEAP_15_Standard_mean - DEAP_15_Standard_std, DEAP_15_Standard_mean + DEAP_15_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_15_Logical_mean, linewidth=1, label="15-deep Logical")
plt.fill_between(axis_x_250, DEAP_15_Logical_mean - DEAP_15_Logical_std, DEAP_15_Logical_mean + DEAP_15_Logical_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Logical_15.png", bbox_inches='tight')

plt.figure(10)
plt.plot(axis_x_250, DEAP_21_Standard_mean, linewidth=1, label="21-deep Standard")
plt.fill_between(axis_x_250, DEAP_21_Standard_mean - DEAP_21_Standard_std, DEAP_21_Standard_mean + DEAP_21_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_21_Logical_mean, linewidth=1, label="21-deep Logical")
plt.fill_between(axis_x_250, DEAP_21_Logical_mean - DEAP_21_Logical_std, DEAP_21_Logical_mean + DEAP_21_Logical_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Logical_21.png", bbox_inches='tight')

plt.legend(loc=legend_loc)
plt.show()

# Standard vs Multiplication
plt.figure(11)
plt.plot(axis_x_250, DEAP_4_Standard_mean, linewidth=1, label="4-deep Standard")
plt.fill_between(axis_x_250, DEAP_4_Standard_mean - DEAP_4_Standard_std, DEAP_4_Standard_mean + DEAP_4_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_4_Multiplication_mean, linewidth=1, label="4-deep Multiplication")
plt.fill_between(axis_x_250, DEAP_4_Multiplication_mean - DEAP_4_Multiplication_std, DEAP_4_Multiplication_mean + DEAP_4_Multiplication_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Multiplication_4.png", bbox_inches='tight')

plt.figure(12)
plt.plot(axis_x_250, DEAP_5_Standard_mean, linewidth=1, label="5-deep Standard")
plt.fill_between(axis_x_250, DEAP_5_Standard_mean - DEAP_5_Standard_std, DEAP_5_Standard_mean + DEAP_5_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_5_Multiplication_mean, linewidth=1, label="5-deep Multiplication")
plt.fill_between(axis_x_250, DEAP_5_Multiplication_mean - DEAP_5_Multiplication_std, DEAP_5_Multiplication_mean + DEAP_5_Multiplication_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Multiplication_5.png", bbox_inches='tight')

plt.legend(loc=legend_loc)
plt.show()

# Standard vs Modified
plt.figure(13)
plt.plot(axis_x_250, DEAP_4_Standard_mean, linewidth=1, label="4-deep Standard")
plt.fill_between(axis_x_250, DEAP_4_Standard_mean - DEAP_4_Standard_std, DEAP_4_Standard_mean + DEAP_4_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_4_Modified_5_mean, linewidth=1, label="4-deep Modified 0.5")
plt.fill_between(axis_x_250, DEAP_4_Modified_5_mean - DEAP_4_Modified_5_std, DEAP_4_Modified_5_mean + DEAP_4_Modified_5_std, alpha=.3)
plt.plot(axis_x_250, DEAP_4_Modified_25_mean, linewidth=1, label="4-deep Modified 0.25")
plt.fill_between(axis_x_250, DEAP_4_Modified_25_mean - DEAP_4_Modified_25_std, DEAP_4_Modified_25_mean + DEAP_4_Modified_25_std, alpha=.3)
plt.plot(axis_x_250, DEAP_4_Modified_125_mean, linewidth=1, label="4-deep Modified 0.125")
plt.fill_between(axis_x_250, DEAP_4_Modified_125_mean - DEAP_4_Modified_125_std, DEAP_4_Modified_125_mean + DEAP_4_Modified_125_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Modified_4.png", bbox_inches='tight')

plt.figure(14)
plt.plot(axis_x_250, DEAP_5_Standard_mean, linewidth=1, label="5-deep Standard")
plt.fill_between(axis_x_250, DEAP_5_Standard_mean - DEAP_5_Standard_std, DEAP_5_Standard_mean + DEAP_5_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_5_Modified_5_mean, linewidth=1, label="5-deep Modified 0.5")
plt.fill_between(axis_x_250, DEAP_5_Modified_5_mean - DEAP_5_Modified_5_std, DEAP_5_Modified_5_mean + DEAP_5_Modified_5_std, alpha=.3)
plt.plot(axis_x_250, DEAP_5_Modified_25_mean, linewidth=1, label="5-deep Modified 0.25")
plt.fill_between(axis_x_250, DEAP_5_Modified_25_mean - DEAP_5_Modified_25_std, DEAP_5_Modified_25_mean + DEAP_5_Modified_25_std, alpha=.3)
plt.plot(axis_x_250, DEAP_5_Modified_125_mean, linewidth=1, label="5-deep Modified 0.125")
plt.fill_between(axis_x_250, DEAP_5_Modified_125_mean - DEAP_5_Modified_125_std, DEAP_5_Modified_125_mean + DEAP_5_Modified_125_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Modified_5.png", bbox_inches='tight')

plt.figure(15)
plt.plot(axis_x_250, DEAP_6_Standard_mean, linewidth=1, label="6-deep Standard")
plt.fill_between(axis_x_250, DEAP_6_Standard_mean - DEAP_6_Standard_std, DEAP_6_Standard_mean + DEAP_6_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_6_Modified_5_mean, linewidth=1, label="6-deep Modified 0.5")
plt.fill_between(axis_x_250, DEAP_6_Modified_5_mean - DEAP_6_Modified_5_std, DEAP_6_Modified_5_mean + DEAP_6_Modified_5_std, alpha=.3)
plt.plot(axis_x_250, DEAP_6_Modified_25_mean, linewidth=1, label="6-deep Modified 0.25")
plt.fill_between(axis_x_250, DEAP_6_Modified_25_mean - DEAP_6_Modified_25_std, DEAP_6_Modified_25_mean + DEAP_6_Modified_25_std, alpha=.3)
plt.plot(axis_x_250, DEAP_6_Modified_125_mean, linewidth=1, label="6-deep Modified 0.125")
plt.fill_between(axis_x_250, DEAP_6_Modified_125_mean - DEAP_6_Modified_125_std, DEAP_6_Modified_125_mean + DEAP_6_Modified_125_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Modified_6.png", bbox_inches='tight')

plt.figure(16)
plt.plot(axis_x_250, DEAP_15_Standard_mean, linewidth=1, label="15-deep Standard")
plt.fill_between(axis_x_250, DEAP_15_Standard_mean - DEAP_15_Standard_std, DEAP_15_Standard_mean + DEAP_15_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_15_Modified_5_mean, linewidth=1, label="15-deep Modified 0.5")
plt.fill_between(axis_x_250, DEAP_15_Modified_5_mean - DEAP_15_Modified_5_std, DEAP_15_Modified_5_mean + DEAP_15_Modified_5_std, alpha=.3)
plt.plot(axis_x_250, DEAP_15_Modified_25_mean, linewidth=1, label="15-deep Modified 0.25")
plt.fill_between(axis_x_250, DEAP_15_Modified_25_mean - DEAP_15_Modified_25_std, DEAP_15_Modified_25_mean + DEAP_15_Modified_25_std, alpha=.3)
plt.plot(axis_x_250, DEAP_15_Modified_125_mean, linewidth=1, label="15-deep Modified 0.125")
plt.fill_between(axis_x_250, DEAP_15_Modified_125_mean - DEAP_15_Modified_125_std, DEAP_15_Modified_125_mean + DEAP_15_Modified_125_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Modified_15.png", bbox_inches='tight')

plt.figure(17)
plt.plot(axis_x_250, DEAP_21_Standard_mean, linewidth=1, label="21-deep Standard")
plt.fill_between(axis_x_250, DEAP_21_Standard_mean - DEAP_21_Standard_std, DEAP_21_Standard_mean + DEAP_21_Standard_std, alpha=.3)
plt.plot(axis_x_250, DEAP_21_Modified_5_mean, linewidth=1, label="21-deep Modified 0.5")
plt.fill_between(axis_x_250, DEAP_21_Modified_5_mean - DEAP_21_Modified_5_std, DEAP_21_Modified_5_mean + DEAP_21_Modified_5_std, alpha=.3)
plt.plot(axis_x_250, DEAP_21_Modified_25_mean, linewidth=1, label="21-deep Modified 0.25")
plt.fill_between(axis_x_250, DEAP_21_Modified_25_mean - DEAP_21_Modified_25_std, DEAP_21_Modified_25_mean + DEAP_21_Modified_25_std, alpha=.3)
plt.plot(axis_x_250, DEAP_21_Modified_125_mean, linewidth=1, label="21-deep Modified 0.125")
plt.fill_between(axis_x_250, DEAP_21_Modified_125_mean - DEAP_21_Modified_125_std, DEAP_21_Modified_125_mean + DEAP_21_Modified_125_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=legend_loc)
plt.savefig("../Plotting/Sequence Classification/Standard_vs_Modified_21.png", bbox_inches='tight')

plt.legend(loc=legend_loc)
plt.show()