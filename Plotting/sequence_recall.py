import matplotlib.pyplot as plt
import pickle
import os
import csv

local_dir = os.path.dirname(__file__)
axis_x_250 = list(range(0, 251))
axis_x_500 = list(range(0, 500))
y_label = "Success Percentage"
x_label = "Training Generations"

'''
    DEAP Reports - Standard
'''

# 4 Deep Report
plt.figure(1)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/4-deep-report/4-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_4_Standard.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(2)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/5-deep-report/5-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_5_Standard.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(3)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/6-deep-report/6-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_6_Standard.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(4)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/15-deep-report/15-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_15_Standard.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(5)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/21-deep-report/21-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_21_Standard.png", bbox_inches='tight')
plt.show()

'''
    DEAP Reports - Logical
'''

# 4 Deep Report
plt.figure(6)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/4-deep-report/4-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_4_Logic.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(7)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/5-deep-report/5-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_5_Logic.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(8)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/6-deep-report/6-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_6_Logic.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(9)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/15-deep-report/15-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_15_Logic.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(10)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/21-deep-report/21-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_21_Logic.png", bbox_inches='tight')
plt.show()

'''
    DEAP Reports - Multiplication
'''

# 4 Deep Report
plt.figure(11)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/4-deep-report/4-progress_report_mul' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_4_multiplication.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(12)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/5-deep-report/5-progress_report_mul' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_5_multiplication.png", bbox_inches='tight')
plt.show()

'''
    DEAP Reports - Modified
'''
# 4 Deep Report
plt.figure(13)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/4-deep-report/4-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_4_mod.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(14)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/5-deep-report/5-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_5_mod.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(15)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/6-deep-report/6-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_6_mod.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(16)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/15-deep-report/15-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_15_mod.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(17)
for i in range(1,21):
    path = os.path.join(local_dir, '../DEAP/Sequence Recall/21-deep-report/21-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_21_mod.png", bbox_inches='tight')
plt.show()


'''
    NEAT Reports
'''

# 4-deep Report
plt.figure(18)
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Recall/4-deep-report/4-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
            axis_x = list(range(0, len(info)))
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Recall/NEAT_4_Standard.png", bbox_inches='tight')

# 5-deep Report
plt.figure(19)
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Recall/5-deep-report/5-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
            axis_x = list(range(0, len(info)))
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Recall/NEAT_5_Standard.png", bbox_inches='tight')

# 6-deep Report
plt.figure(20)
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Recall/6-deep-report/6-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
            axis_x = list(range(0, len(info)))
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Recall/NEAT_6_Standard.png", bbox_inches='tight')

# 15-deep Report
plt.figure(21)
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Recall/15-deep-report/15-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
            axis_x = list(range(0, len(info)))
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Recall/NEAT_15_Standard.png", bbox_inches='tight')

# 21-deep Report
plt.figure(22)
for i in range(1,21):
    path = os.path.join(local_dir, '../NEAT/Sequence Recall/21-deep-report/21-progress_report' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            fitness = row[0].split()
            info.append(float(fitness[0]))
            axis_x = list(range(0, len(info)))
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Recall/NEAT_21_Standard.png", bbox_inches='tight')

plt.show()