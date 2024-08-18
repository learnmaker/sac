import numpy as np

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
array3 = np.array([7, 8, 9])
array4 = np.array([10, 11, 12])

arrays = [array1, array2, array3, array4]
print(arrays)

matrix = np.hstack(arrays)
print(matrix)