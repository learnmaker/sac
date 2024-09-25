import random
import os, sys, inspect
# set parent directory as sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from tool.data_loader import load_data

# 序列A的规则
sequence_a_rules = {}

# 序列B的时间段规则
sequence_b_rules = {}

def init(task_num):
    for i in range(0, task_num, 2):
        sequence_a_rules[i] = random.sample(range(0, task_num), task_num//3)
        
    sequence_b_rules['morning'] = [0, task_num//3]
    sequence_b_rules['afternoon'] = [task_num//3 + 1, 2*task_num//3]
    sequence_b_rules['evening'] = [2*task_num//3 + 1, task_num-1]
    
# 生成所有的任务请求
def generate_request(server_num, ud_num, task_num):
    # 如果有对应任务请求文件，则直接跳过
    flag = True
    for server in range(server_num):
        for ud in range(ud_num):
            if os.path.exists("./mydata/temp/server"+str(server+1)+"_ud"+str(ud+1)+"_samples"+str(task_num)+".csv") and os.path.isfile("./mydata/temp/server"+str(server+1)+"_ud"+str(ud+1)+"_samples"+str(task_num)+".csv"):
                continue
            else:
                flag = False
                print("生成任务请求文件中...")
                break
    if flag:
        print("存在对应任务请求文件")
        return
    
    init(task_num)
    task = range(task_num)
    length = 1000000
    # 给所有用户设备添加任务列表
    for server in range(server_num):
        for ud in range(ud_num):
            sequence = []
            current_task = random.choice(task)
            for step in range(length):
                sequence.append(current_task)
                # 规则A
                if random.random()>0.5:
                    if current_task in sequence_a_rules:
                        next_tasks = sequence_a_rules[current_task]
                        current_task = random.choice(next_tasks)
                    else:
                        current_task = random.choice(task)
                # 规则B
                else:
                    if 0 <= step % 24 <8:
                        tasks_pool = sequence_b_rules['morning']
                    elif 8 <= step % 24 <16:
                        tasks_pool = sequence_b_rules['afternoon']
                    else:
                        tasks_pool = sequence_b_rules['evening']
                    current_task = random.choice(tasks_pool)
            # 保存该服务器的该用户设备的任务请求
            pd.DataFrame(np.array(sequence)).to_csv("./mydata/temp/server"+str(server+1)+"_ud"+str(ud+1)+"_samples"+str(task_num)+".csv", header=False, index=False)
           
    return
if __name__ == "__main__":
    init(8)
    print(sequence_a_rules)
    print(sequence_b_rules)