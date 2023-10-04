import numpy as np
import matplotlib.pyplot as plt

winding_min = 1
winding_max = 100
current_min = 0
current_max = 2
sample_points = 100

# read from data.csv
data = np.loadtxt('data.csv', delimiter=',')

plt.imshow(data, cmap='hot', interpolation='nearest', extent=[current_min, current_max, winding_max, winding_min], aspect='auto')
plt.xlabel('Current (A)')
plt.ylabel('Winding (turns)')

# for each row, find the point with value closest to -15
for i in range(len(data)):
    min_value = 1000
    min_index = 0
    row = data[i]
    for j in range(len(row)):
        if abs(row[j] - -15) < min_value:
            min_value = abs(row[j] - -15)
            min_index = j
    plt.plot(current_max* (i / sample_points), min_index, 'bo')
    print('Winding: ' + str(min_index) + ' Current: ' + str(current_max* (i / sample_points)))


plt.colorbar()
plt.show()