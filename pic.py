import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# 设置字体为中文字体
# matplotlib.rcParams['font.family'] = 'SimHei'

# 读取CSV文件
filename = input("文件名称：")
data1 = pd.read_csv("runs/"+ filename + "/update_parameters.csv")
data2 = pd.read_csv("runs/"+ filename + "/episode_rewards.csv")
data3 = pd.read_csv("runs/"+ filename + "/eval.csv")

if not os.path.exists("runs/"+ filename + "/pic/"):
    os.makedirs("runs/"+ filename + "/pic/")

# data1
headers1 = data1.columns.tolist()
parameters = ["critic_loss", "actor_loss", "entropy_loss", "alpha"]
for index,parameter in enumerate(parameters):
    
    for i in range(len(headers1)//4):
        x = list(range(data1.shape[0]))
        y = data1.iloc[:, index+i*4].tolist()
        
        plt.plot(x, y, label=headers1[index+i*4])
        
    # 添加标题、标签和图例
    plt.title(parameter + ' for agent')
    plt.xlabel('Number of updates')
    plt.ylabel(parameter)
    plt.legend()

    plt.savefig("runs/"+ filename + "/pic/" + parameter + '随采样更新变化图.png')
    plt.clf()  # 清除整个图表
    
    
# data2
headers2 = data2.columns.tolist()

# 遍历每列并绘制折线图
for i in range(len(headers2)-1):
    x = list(range(data2.shape[0]))
    y = data2.iloc[:, i].tolist()
    
    plt.plot(x, y, label=headers2[i])

# 添加标题、标签和图例
plt.title('episode_rewards for agent')
plt.xlabel('episode')
plt.ylabel('Rewards')
plt.legend()

plt.savefig("runs/"+ filename + "/pic/" + '各agent的episode_rewards.png')
plt.clf()  # 清除整个图表

x = list(range(data2.shape[0]))
y = data2.iloc[:, -1].tolist()
plt.plot(x, y, label=headers2[-1])
plt.title('total_reward for system')
plt.xlabel('episode')
plt.ylabel('total_reward')
plt.legend()

plt.savefig("runs/"+ filename + "/pic/" + '系统总体reward.png')
plt.clf()  # 清除整个图表

# data3
headers3 = data3.columns.tolist()
items = ["test_reword", "trans_cost", "comp_cost", "total_cost"]
for index, item in enumerate(items):
    x = list(range(data3.shape[0]))
    y = data3.iloc[:, index].tolist()
    
    plt.plot(x, y, label=headers3[index])
    plt.title(item + ' for system')
    plt.xlabel('Number of test')
    plt.ylabel(item)
    plt.legend()

    plt.savefig("runs/"+ filename + "/pic/" + '{}评估结果.png'.format(item))
    plt.clf()  # 清除整个图表