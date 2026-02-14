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
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/4_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_4_Standard.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(2)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/5_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_5_Standard.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(3)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/6_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_6_Standard.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(4)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/15_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_15_Standard.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(5)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/21_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_21_Standard.png", bbox_inches='tight')
plt.show()

'''
    DEAP Reports - Logical
'''

# 4 Deep Report
plt.figure(6)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/4_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_4_Logic.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(7)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/5_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_5_Logic.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(8)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/6_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_6_Logic.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(9)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/15_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_15_Logic.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(10)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/21_report_log')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_21_Logic.png", bbox_inches='tight')
plt.show()

'''
    DEAP Reports - Multiplication
'''

# 4 Deep Report
plt.figure(11)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/4_report_mul')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_4_multiplication.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(12)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/5_report_mul')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_5_multiplication.png", bbox_inches='tight')
plt.show()

'''
    DEAP Reports - Modified
'''
# 4 Deep Report
plt.figure(13)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/4_report_mod_5')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_4_5.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(14)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/5_report_mod_5')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_5_5.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(15)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/6_report_mod_5')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_6_5.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(16)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/15_report_mod_5')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_15_5.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(17)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/21_report_mod_5')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_21_5.png", bbox_inches='tight')
plt.show()

# 4 Deep Report
plt.figure(18)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/4_report_mod_25')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_4_25.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(19)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/5_report_mod_25')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_5_25.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(20)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/6_report_mod_25')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_6_25.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(21)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/15_report_mod_25')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_15_25.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(22)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/21_report_mod_25')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_21_25.png", bbox_inches='tight')
plt.show()

# 4 Deep Report
plt.figure(23)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/4_report_mod_125')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_4_125.png", bbox_inches='tight')

# 5 Deep Report
plt.figure(24)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/5_report_mod_125')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_5_125.png", bbox_inches='tight')

# 6 Deep Report
plt.figure(25)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/6_report_mod_125')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_6_125.png", bbox_inches='tight')

# 15 Deep Report
plt.figure(26)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/15_report_mod_125')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_15_125.png", bbox_inches='tight')

# 21 Deep Report
plt.figure(27)
path = os.path.join(local_dir, '../DEAP/Sequence Classification/reports/21_report_mod_125')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    plt.plot(axis_x_250, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.savefig("../Plotting/Sequence Classification/DEAP_21_125.png", bbox_inches='tight')
plt.show()


'''
    NEAT Reports
'''

# 4-deep Report
plt.figure(28)
path = os.path.join(local_dir, '../NEAT/Sequence Classification/reports/4_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Classification/NEAT_4_Standard.png", bbox_inches='tight')

# 5-deep Report
plt.figure(29)
path = os.path.join(local_dir, '../NEAT/Sequence Classification/reports/5_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Classification/NEAT_5_Standard.png", bbox_inches='tight')

# 6-deep Report
plt.figure(30)
path = os.path.join(local_dir, '../NEAT/Sequence Classification/reports/6_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Classification/NEAT_6_Standard.png", bbox_inches='tight')

# 15-deep Report
plt.figure(31)
path = os.path.join(local_dir, '../NEAT/Sequence Classification/reports/15_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Classification/NEAT_15_Standard.png", bbox_inches='tight')

# 21-deep Report
plt.figure(32)
path = os.path.join(local_dir, '../NEAT/Sequence Classification/reports/21_report_std')
with open(path, 'rb') as f:
    data = pickle.load(f)
for info in data.values():
    info.extend([info[-1] for _ in range(500 - len(info))])
    plt.plot(axis_x_500, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.savefig("../Plotting/Sequence Classification/NEAT_21_Standard.png", bbox_inches='tight')

plt.show()