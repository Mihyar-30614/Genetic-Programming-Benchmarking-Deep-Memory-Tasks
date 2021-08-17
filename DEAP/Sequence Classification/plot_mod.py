import matplotlib.pyplot as plt
import pickle
import os

local_dir = os.path.dirname(__file__)
axis_x = list(range(0, 251))
ngen = len(axis_x) - 1
y_label = "Success Percentage"
x_label = "Training Generations"

# 4 Deep Report
for i in range(1,11):
    path = os.path.join(local_dir, '4-deep-report/4-progress_report_0.5_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.title(str(ngen) + "-Gen 4-deep Report using 0.5")
plt.show()

# 4 Deep Report
for i in range(1,11):
    path = os.path.join(local_dir, '4-deep-report/4-progress_report_0.25_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.title(str(ngen) + "-Gen 4-deep Report using 0.25")
plt.show()

# 4 Deep Report
for i in range(1,11):
    path = os.path.join(local_dir, '4-deep-report/4-progress_report_0.125_' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.title(str(ngen) + "-Gen 4-deep Report using 0.125")
plt.show()

# 5 Deep Report
for i in range(1,21):
    path = os.path.join(local_dir, '5-deep-report/5-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.title(str(ngen) + "-Gen 5-deep Report")
plt.show()

# 6 Deep Report
for i in range(1,21):
    path = os.path.join(local_dir, '6-deep-report/6-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.title(str(ngen) + "-Gen 6-deep Report")
plt.show()

# 15 Deep Report
for i in range(1,21):
    path = os.path.join(local_dir, '15-deep-report/15-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.title(str(ngen) + "-Gen 15-deep Report")
plt.show()

# 21 Deep Report
for i in range(1,21):
    path = os.path.join(local_dir, '21-deep-report/21-progress_report_mod' + str(i))
    with open(path, 'rb') as f:
        info = pickle.load(f)
    plt.plot(axis_x, info, linewidth=1)

plt.ylabel(y_label)
plt.xlabel(x_label)
plt.title(str(ngen) + "-Gen 21-deep Report")
plt.show()