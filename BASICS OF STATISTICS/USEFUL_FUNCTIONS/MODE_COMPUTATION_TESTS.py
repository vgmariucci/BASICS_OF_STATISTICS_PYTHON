import numpy as np

#create NumPy array of values with only one mode
x = np.array([2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 7])

#find unique values in array along with their counts
vals, counts = np.unique(x, return_counts=True)

#find mode
mode_value = np.argwhere(counts == np.max(counts))

#print list of modes
print("\n", vals[mode_value].flatten().tolist())



#find how often mode occurs
print("\n", np.max(counts))


#create NumPy array of values with multiple modes
x = np.array([2, 2, 2, 3, 4, 4, 4, 5, 5, 5, 7])

#find unique values in array along with their counts
vals, counts = np.unique(x, return_counts=True)

#find mode
mode_value = np.argwhere(counts == np.max(counts))

#print list of modes
print("\n", vals[mode_value].flatten().tolist())



#find how often mode occurs
print("\n", np.max(counts))


