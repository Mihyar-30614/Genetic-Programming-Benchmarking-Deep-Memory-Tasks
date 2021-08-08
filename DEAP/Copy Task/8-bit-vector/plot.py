import matplotlib.pyplot as plt
import pickle
import os

local_dir = os.path.dirname(__file__)

# 4 Deep Report
rpt_1_path = os.path.join(local_dir, '8-bit-vector-report/8-progress_report1')
rpt_2_path = os.path.join(local_dir, '8-bit-vector-report/8-progress_report2')
rpt_3_path = os.path.join(local_dir, '8-bit-vector-report/8-progress_report3')
rpt_4_path = os.path.join(local_dir, '8-bit-vector-report/8-progress_report4')
rpt_5_path = os.path.join(local_dir, '8-bit-vector-report/8-progress_report5')
rpt_6_path = os.path.join(local_dir, '8-bit-vector-report/8-progress_report6')
rpt_7_path = os.path.join(local_dir, '8-bit-vector-report/8-progress_report7')
rpt_8_path = os.path.join(local_dir, '8-bit-vector-report/8-progress_report8')
rpt_9_path = os.path.join(local_dir, '8-bit-vector-report/8-progress_report9')
rpt_10_path = os.path.join(local_dir, '8-bit-vector-report/8-progress_report10')

# Import info
with open(rpt_1_path, 'rb') as f:
    rpt_1 = pickle.load(f)

with open(rpt_2_path, 'rb') as f:
    rpt_2 = pickle.load(f)

with open(rpt_3_path, 'rb') as f:
    rpt_3 = pickle.load(f)

with open(rpt_4_path, 'rb') as f:
    rpt_4 = pickle.load(f)

with open(rpt_5_path, 'rb') as f:
    rpt_5 = pickle.load(f)

with open(rpt_6_path, 'rb') as f:
    rpt_6 = pickle.load(f)

with open(rpt_7_path, 'rb') as f:
    rpt_7 = pickle.load(f)

with open(rpt_8_path, 'rb') as f:
    rpt_8 = pickle.load(f)

with open(rpt_9_path, 'rb') as f:
    rpt_9 = pickle.load(f)

with open(rpt_10_path, 'rb') as f:
    rpt_10 = pickle.load(f)

axis_x = list(range(0, 251))
ngen = len(axis_x) - 1
plt.plot(axis_x, rpt_1, linewidth=1)
plt.plot(axis_x, rpt_2, linewidth=1)
plt.plot(axis_x, rpt_3, linewidth=1)
plt.plot(axis_x, rpt_4, linewidth=1)
plt.plot(axis_x, rpt_5, linewidth=1)
plt.plot(axis_x, rpt_6, linewidth=1)
plt.plot(axis_x, rpt_7, linewidth=1)
plt.plot(axis_x, rpt_8, linewidth=1)
plt.plot(axis_x, rpt_9, linewidth=1)
plt.plot(axis_x, rpt_10, linewidth=1)
plt.ylabel('Success Percentage')
plt.xlabel('Training Generations')
plt.title(str(ngen) + "-Gen 8-bit Report")
plt.show()