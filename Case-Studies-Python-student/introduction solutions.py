import numpy as np
from matplotlib import pyplot as plt

#========================
#
# a  = np.array([[np.random.rand() for b in range(10)] for b in range(1000)])
# print(a)
# plt.hist(a)
# plt.show()
#
# print("===================")
# print(np.shape(a))

spike_threshold = 4
resting_voltage = 1
input_current_per_time = 0.1
end_time = 1000
start_time = 0
time_frame = np.array([b for b in range(start_time,end_time)])

voltage_over_time = np.array([((b*input_current_per_time) % (spike_threshold-resting_voltage)) + resting_voltage for b in time_frame])

plt.plot(time_frame,voltage_over_time)
plt.show()
