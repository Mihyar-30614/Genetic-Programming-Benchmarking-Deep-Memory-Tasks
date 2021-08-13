import matplotlib.pyplot as plt
import pickle
import os

local_dir = os.path.dirname(__file__)
axis_x = list(range(0, 251))
ngen = len(axis_x) - 1
y_label = "Success Percentage"
x_label = "Training Generations"

# 8-bit Report
for i in range(1,21):
    path = os.path.join(local_dir, '8-bit-report/8-progress_report' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.title(str(ngen) + "-Gen 8-bit Report")
plt.show()