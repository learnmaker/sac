import numpy as np

# 假设 padding 是一个 (3, 17) 形状的数组
padding = np.zeros((3, 17))

# 假设 state_sequence[index] 是一个 (2, 17) 形状的数组
state_sequence = [np.ones((2, 17))]  # 示例数据
index = 0  # 选择索引

# 直接使用 np.concatenate 来垂直堆叠两个数组
state_sequence_new = np.concatenate((padding, state_sequence[index]), axis=0)
print(state_sequence_new)