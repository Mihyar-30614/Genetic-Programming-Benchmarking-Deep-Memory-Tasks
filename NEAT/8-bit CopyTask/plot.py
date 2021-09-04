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
for i in range(1,21):
    path = os.path.join(local_dir, '8-bit-report/fitness_history' + str(i) + '.csv')
    info = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            info.append(float(*row))
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.title(str(ngen) + "-Gen 8-bit Report")
plt.xlim([0, 500])
plt.ylim([0, 100])
plt.show()