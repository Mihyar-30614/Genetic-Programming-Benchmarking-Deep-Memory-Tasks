import matplotlib.pyplot as plt
import pickle
import os
import csv

local_dir = os.path.dirname(__file__)
axis_x_250 = list(range(251))
axis_x_500 = list(range(500))
y_label = "Success Percentage"
x_label = "Training Generations"

'''
    DEAP Reports - Standard
'''

# 4 Deep Report
plt.figure(1)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/4_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_4_Standard.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(2)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/5_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_5_Standard.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(3)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/6_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_6_Standard.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(4)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/15_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_15_Standard.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(5)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/21_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
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
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/4_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_4_Logic.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(7)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/5_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_5_Logic.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(8)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/6_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_6_Logic.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(9)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/15_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_15_Logic.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(10)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/21_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
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
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/4_report_mul')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_4_multiplication.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(12)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/5_report_mul')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
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
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/4_report_mod')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_4_mod.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(14)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/5_report_mod')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_5_mod.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(15)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/6_report_mod')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_6_mod.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(16)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/15_report_mod')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Recall/DEAP_15_mod.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(17)
path = os.path.join(local_dir, '../DEAP/Sequence Recall/reports/21_report_mod')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
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
path = os.path.join(local_dir, '../NEAT/Sequence Recall/reports/4_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Recall/NEAT_4_Standard.png", bbox_inches='tight')

# 5-deep Report
plt.figure(19)
path = os.path.join(local_dir, '../NEAT/Sequence Recall/reports/5_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Recall/NEAT_5_Standard.png", bbox_inches='tight')

# 6-deep Report
plt.figure(20)
path = os.path.join(local_dir, '../NEAT/Sequence Recall/reports/6_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Recall/NEAT_6_Standard.png", bbox_inches='tight')

# 15-deep Report
plt.figure(21)
path = os.path.join(local_dir, '../NEAT/Sequence Recall/reports/15_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Recall/NEAT_15_Standard.png", bbox_inches='tight')

# 21-deep Report
plt.figure(22)
path = os.path.join(local_dir, '../NEAT/Sequence Recall/reports/21_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Recall/NEAT_21_Standard.png", bbox_inches='tight')

plt.show()