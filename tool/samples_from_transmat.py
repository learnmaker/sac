import os, sys, inspect
# set parent directory as sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from tool.data_loader import load_data


def generate_sample(cur_state, xk, A):
    return np.random.choice(xk, 1, p=A[cur_state, :])

def generate_request(server_num, ud_num, task_num, maxp):
    # 选择任务数、最大转移概率
    trans_mat = load_data('./mydata/trans'+str(task_num)+'_maxp'+str(maxp)+'.csv')
    A = np.array(trans_mat)
    xk = np.arange(len(A))
    initial_state = 0
    sample_len = 1000000
    output = [-1 for i in range(sample_len)]
    output[0] = initial_state
    for i in range(1, sample_len):
        output[i] = generate_sample(output[i - 1], xk, A)[0]
    for server in range(server_num):
        for ud in range(ud_num):
    
    pd.DataFrame(np.array(output)).to_csv("./mydata/samples6_maxp70.csv", header=False, index=False)       

if __name__ == '__main__':
    # 选择任务数、最大转移概率
    trans_mat = load_data('./data/trans6_maxp70.csv')
    A = np.array(trans_mat)
    print(np.sum(A, 1))
    xk = np.arange(len(A))
    initial_state = 0
    sample_len = 1000000
    output = [-1 for i in range(sample_len)]
    output[0] = initial_state
    for i in range(1, sample_len):
        output[i] = generate_sample(output[i - 1], xk, A)[0]

    pd.DataFrame(np.array(output)).to_csv("./mydata/samples6_maxp70.csv", header=False, index=False)
