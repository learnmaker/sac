import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('data/samples4_maxp70.csv', header=None)

# 将数据转换为列表
values = data.iloc[:100, 0].tolist()

# 创建x轴索引，假设数据点的顺序即为时间序列的顺序
x = list(range(len(values)))

# 绘制折线图
plt.figure(figsize=(10, 5))
plt.plot(x, values)
plt.title('折线图')
plt.xlabel('时间点')
plt.ylabel('数值')

# 显示图形
plt.show()