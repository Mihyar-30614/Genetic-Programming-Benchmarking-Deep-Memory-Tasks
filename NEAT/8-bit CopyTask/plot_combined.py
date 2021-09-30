import matplotlib.pyplot as plt
import csv
import os
import numpy as np

local_dir = os.path.dirname(__file__)
axis_x = list(range(0, 500))
ngen = len(axis_x)
y_label = "Success Percentage"
x_label = "Training Generations"

# 8-bit Report
plt.figure(1)
standard_info = []
for i in range(1,21):
    path = os.path.join(local_dir, '8-bit-report/fitness_history' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            info.append(float(*row))
    standard_info.append(info)

standard_mean = np.mean(standard_info, axis=0)
standard_std = np.std(standard_info, axis=0)

plt.plot(axis_x, standard_mean, linewidth=1, label="Standard")
plt.fill_between(axis_x, standard_mean - standard_std, standard_mean + standard_std, alpha=.3)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.title(str(ngen) + "-Gen 8-bit Report")
plt.legend(loc="lower right")
plt.savefig("standard_combined.png", bbox_inches='tight')
plt.show()