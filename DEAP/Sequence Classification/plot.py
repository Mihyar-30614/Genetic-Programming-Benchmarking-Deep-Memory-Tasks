import matplotlib.pyplot as plt
import pickle
import os

local_dir = os.path.dirname(__file__)

# 1K all report
deep_4 = os.path.join(local_dir, '1k-training-report/4-progress_report')
deep_5 = os.path.join(local_dir, '1k-training-report/5-progress_report')
deep_6 = os.path.join(local_dir, '1k-training-report/6-progress_report')
deep_15 = os.path.join(local_dir, '1k-training-report/15-progress_report')
deep_21 = os.path.join(local_dir, '1k-training-report/21-progress_report')

# Import info
with open(deep_4, 'rb') as f:
    progress_report_4 = pickle.load(f)

with open(deep_5, 'rb') as f:
    progress_report_5 = pickle.load(f)

with open(deep_6, 'rb') as f:
    progress_report_6 = pickle.load(f)

with open(deep_15, 'rb') as f:
    progress_report_15 = pickle.load(f)

with open(deep_15, 'rb') as f:
    progress_report_21 = pickle.load(f)

axis_x = list(range(0, 1001))
line1, = plt.plot(axis_x, progress_report_4, label="4-deep", linewidth=1)
line2, = plt.plot(axis_x, progress_report_5, label="5-deep", linewidth=1)
line3, = plt.plot(axis_x, progress_report_6, label="6-deep", linewidth=1)
line4, = plt.plot(axis_x, progress_report_15, label="15-deep", linewidth=1)
line5, = plt.plot(axis_x, progress_report_21, label="21-deep", linewidth=1)
plt.ylabel('Success Percentage')
plt.xlabel('Training Generations')
plt.legend(handles=[line1, line2, line3, line4, line5], loc="lower right")
plt.title("1K-Gen All Report")
plt.show()

# 1K all reports zoomed in
progress_report_4 = progress_report_4[:100]
progress_report_5 = progress_report_5[:100]
progress_report_6 = progress_report_6[:100]
progress_report_15 = progress_report_15[:100]
progress_report_21 = progress_report_21[:100]

axis_x = list(range(0, 100))
line1, = plt.plot(axis_x, progress_report_4, label="4-deep", linewidth=1)
line2, = plt.plot(axis_x, progress_report_5, label="5-deep", linewidth=1)
line3, = plt.plot(axis_x, progress_report_6, label="6-deep", linewidth=1)
line4, = plt.plot(axis_x, progress_report_15, label="15-deep", linewidth=1)
line5, = plt.plot(axis_x, progress_report_21, label="21-deep", linewidth=1)
plt.ylabel('Success Percentage')
plt.xlabel('Training Generations')
plt.legend(handles=[line1, line2, line3, line4, line5], loc="lower right")
plt.title("1K-Gen All Report Zoom in")
plt.show()

# 4 Deep Report
rpt_1_path = os.path.join(local_dir, '4-deep-report/4-progress_report1')
rpt_2_path = os.path.join(local_dir, '4-deep-report/4-progress_report2')
rpt_3_path = os.path.join(local_dir, '4-deep-report/4-progress_report3')
rpt_4_path = os.path.join(local_dir, '4-deep-report/4-progress_report4')
rpt_5_path = os.path.join(local_dir, '4-deep-report/4-progress_report5')
rpt_6_path = os.path.join(local_dir, '4-deep-report/4-progress_report6')
rpt_7_path = os.path.join(local_dir, '4-deep-report/4-progress_report7')
rpt_8_path = os.path.join(local_dir, '4-deep-report/4-progress_report8')
rpt_9_path = os.path.join(local_dir, '4-deep-report/4-progress_report9')
rpt_10_path = os.path.join(local_dir, '4-deep-report/4-progress_report10')

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
plt.title(str(ngen) + "-Gen 4-deep Report")
plt.show()

# 5 Deep Report
rpt_1_path = os.path.join(local_dir, '5-deep-report/5-progress_report1')
rpt_2_path = os.path.join(local_dir, '5-deep-report/5-progress_report2')
rpt_3_path = os.path.join(local_dir, '5-deep-report/5-progress_report3')
rpt_4_path = os.path.join(local_dir, '5-deep-report/5-progress_report4')
rpt_5_path = os.path.join(local_dir, '5-deep-report/5-progress_report5')
rpt_6_path = os.path.join(local_dir, '5-deep-report/5-progress_report6')
rpt_7_path = os.path.join(local_dir, '5-deep-report/5-progress_report7')
rpt_8_path = os.path.join(local_dir, '5-deep-report/5-progress_report8')
rpt_9_path = os.path.join(local_dir, '5-deep-report/5-progress_report9')
rpt_10_path = os.path.join(local_dir, '5-deep-report/5-progress_report10')

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
plt.title(str(ngen) + "-Gen 5-deep Report")
plt.show()

# 6 Deep Report
rpt_1_path = os.path.join(local_dir, '6-deep-report/6-progress_report1')
rpt_2_path = os.path.join(local_dir, '6-deep-report/6-progress_report2')
rpt_3_path = os.path.join(local_dir, '6-deep-report/6-progress_report3')
rpt_4_path = os.path.join(local_dir, '6-deep-report/6-progress_report4')
rpt_5_path = os.path.join(local_dir, '6-deep-report/6-progress_report5')
rpt_6_path = os.path.join(local_dir, '6-deep-report/6-progress_report6')
rpt_7_path = os.path.join(local_dir, '6-deep-report/6-progress_report7')
rpt_8_path = os.path.join(local_dir, '6-deep-report/6-progress_report8')
rpt_9_path = os.path.join(local_dir, '6-deep-report/6-progress_report9')
rpt_10_path = os.path.join(local_dir, '6-deep-report/6-progress_report10')

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
plt.title(str(ngen) + "-Gen 6-deep Report")
plt.show()

# 15 Deep Report
rpt_1_path = os.path.join(local_dir, '15-deep-report/15-progress_report1')
rpt_2_path = os.path.join(local_dir, '15-deep-report/15-progress_report2')
rpt_3_path = os.path.join(local_dir, '15-deep-report/15-progress_report3')
rpt_4_path = os.path.join(local_dir, '15-deep-report/15-progress_report4')
rpt_5_path = os.path.join(local_dir, '15-deep-report/15-progress_report5')
rpt_6_path = os.path.join(local_dir, '15-deep-report/15-progress_report6')
rpt_7_path = os.path.join(local_dir, '15-deep-report/15-progress_report7')
rpt_8_path = os.path.join(local_dir, '15-deep-report/15-progress_report8')
rpt_9_path = os.path.join(local_dir, '15-deep-report/15-progress_report9')
rpt_10_path = os.path.join(local_dir, '15-deep-report/15-progress_report10')

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
plt.title(str(ngen) + "-Gen 15-deep Report")
plt.show()

# 21 Deep Report
rpt_1_path = os.path.join(local_dir, '21-deep-report/21-progress_report1')
rpt_2_path = os.path.join(local_dir, '21-deep-report/21-progress_report2')
rpt_3_path = os.path.join(local_dir, '21-deep-report/21-progress_report3')
rpt_4_path = os.path.join(local_dir, '21-deep-report/21-progress_report4')
rpt_5_path = os.path.join(local_dir, '21-deep-report/21-progress_report5')
rpt_6_path = os.path.join(local_dir, '21-deep-report/21-progress_report6')
rpt_7_path = os.path.join(local_dir, '21-deep-report/21-progress_report7')
rpt_8_path = os.path.join(local_dir, '21-deep-report/21-progress_report8')
rpt_9_path = os.path.join(local_dir, '21-deep-report/21-progress_report9')
rpt_10_path = os.path.join(local_dir, '21-deep-report/21-progress_report10')

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
plt.title(str(ngen) + "-Gen 21-deep Report")
plt.show()