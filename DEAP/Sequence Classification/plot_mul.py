import matplotlib.pyplot as plt
import pickle
import os

local_dir = os.path.dirname(__file__)
axis_x = list(range(0, 251))
ngen = len(axis_x) - 1
y_label = "Success Percentage"
x_label = "Training Generations"

# 4 Deep Report
plt.figure(1)
for i in range(1,21):
    path = os.path.join(local_dir, '4-deep-report/4-progress_report_mul' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 250])
plt.ylim([0, 100])
plt.title(str(ngen) + "-Gen 4-deep Report")

# 5 Deep Report
plt.figure(2)
for i in range(1,21):
    path = os.path.join(local_dir, '5-deep-report/5-progress_report_mul' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.xlim([0, 250])
plt.ylim([0, 100])
plt.title(str(ngen) + "-Gen 5-deep Report")
plt.show()