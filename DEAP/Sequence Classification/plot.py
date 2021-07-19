import matplotlib.pyplot as plt
import pickle

# Import info
with open('4-progress_report', 'rb') as f:
    progress_report_4 = pickle.load(f)

with open('5-progress_report', 'rb') as f:
    progress_report_5 = pickle.load(f)

with open('6-progress_report', 'rb') as f:
    progress_report_6 = pickle.load(f)

with open('15-progress_report', 'rb') as f:
    progress_report_15 = pickle.load(f)

with open('21-progress_report', 'rb') as f:
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
plt.show()


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
plt.show()