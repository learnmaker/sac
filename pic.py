import sys
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
# filename = input("文件名称：")
filename = "2024-08-18_19-50-06_SAC_case3____"
# data1 = pd.read_csv("D:/temp_data/data/"+ filename + "update_parameters.csv")
data2 = pd.read_csv("D:/temp_data/data/"+ filename + "episode_rewards.csv")
# data3 = pd.read_csv("D:/temp_data/data/"+ filename + "eval.csv")

# 获取表头
headers = data2.columns.tolist()

# 遍历每列并绘制折线图
for i in range(len(headers)-1):
    x = list(range(data2.shape[0]))
    y = data2.iloc[:, i].tolist()
    
    plt.plot(x, y, label=headers[i])

# 添加标题、标签和图例
plt.title('episode_rewards for agent')
plt.xlabel('Time steps')
plt.ylabel('Rewards')
plt.legend()

# 显示图形
plt.show()