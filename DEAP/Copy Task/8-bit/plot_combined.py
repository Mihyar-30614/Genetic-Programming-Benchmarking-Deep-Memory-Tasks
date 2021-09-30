import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

local_dir = os.path.dirname(__file__)
axis_x = list(range(0, 251))
ngen = len(axis_x) - 1
y_label = "Success Percentage"
x_label = "Training Generations"

# 8-bit Report
plt.figure(1)
standard_info = []
for i in range(1,21):
    path = os.path.join(local_dir, '8-bit-report/8-progress_report' + str(i))
    
    with open(path, 'rb') as f:
        info = pickle.load(f)
    standard_info.append(info)

standard_mean = np.mean(standard_info, axis=0)
standard_std = np.std(standard_info, axis=0)

plt.plot(axis_x, standard_mean, linewidth=1, label="Standard")
plt.fill_between(axis_x, standard_mean - standard_std, standard_mean + standard_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.title(str(ngen) + "-Gen 8-bit Report")
plt.legend(loc="lower right")
plt.savefig("standard_combined.png", bbox_inches='tight')

# Combined figure
plt.figure(2)
logic_info = []
for i in range(1,21):
    path = os.path.join(local_dir, '8-bit-report/8-progress_report_logic' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    logic_info.append(info)

logic_mean = np.mean(logic_info, axis=0)
logic_std = np.std(logic_info, axis=0)

mod_info = []
for i in range(1,21):
    path = os.path.join(local_dir, '8-bit-report/8-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    mod_info.append(info)

mod_mean = np.mean(mod_info, axis=0)
mod_std = np.std(mod_info, axis=0)

mul_info = []
for i in range(1,21):
    path = os.path.join(local_dir, '8-bit-report/8-progress_report_mul' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    mul_info.append(info)

mul_mean = np.mean(mul_info, axis=0)
mul_std = np.std(mul_info, axis=0)

#  Plot standard
plt.plot(axis_x, standard_mean, linewidth=1, label="Standard")
plt.fill_between(axis_x, standard_mean - standard_std, standard_mean + standard_std, alpha=.3)

# Plot Logical
plt.plot(axis_x, logic_mean, linewidth=1, label="Logical")
plt.fill_between(axis_x, logic_mean - logic_std, logic_mean + logic_std, alpha=.3)

#  Plot modified
plt.plot(axis_x, mod_mean, linewidth=1, label="Modified")
plt.fill_between(axis_x, mod_mean - mod_std, mod_mean + mod_std, alpha=.3)

#  Plot multiplication
plt.plot(axis_x, mul_mean, linewidth=1, label="Multiplication")
plt.fill_between(axis_x, mul_mean - mul_std, mul_mean + mul_std, alpha=.3)

plt.legend(loc="lower right")
plt.savefig("all_combined.png", bbox_inches='tight')
plt.show()