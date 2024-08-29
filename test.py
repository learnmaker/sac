import numpy as np

# 示例数据
array1 = np.array([1, 2, 3])  # 一维向量
array2 = np.array([[4, 5, 6], [7, 8, 9]])  # 二维矩阵

# 创建一个空的二维数组
# encode_data = np.empty((0, len(array1) + len(array2.flatten())), dtype=int)
encode_data = []

# 连接向量和矩阵，并添加到 encode_data 中
connected_data = np.hstack((array1, array2.flatten()))

# 添加到 encode_data
encode_data.append(connected_data)

# 输出结果
print("Resulting array:")
print(encode_data)