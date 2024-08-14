import numpy as np

# 创建一个一维数组
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 将数组重塑为 3x3 的二维数组
reshaped_arr = arr.reshape((3, 3))

print(reshaped_arr)

reshaped_arr = arr.view(-1)
print(reshaped_arr)