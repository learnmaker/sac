import multiprocessing
import time
import numpy as np
import torch
from gym import spaces

def index2ud(index, server_num, ud_num):
    server = index // ud_num
    ud = index - server * ud_num
    return server, ud

if __name__ == '__main__':
    task_lists=[[] for _ in range(6)]
    for i in range(5):
        task_lists[i].append([i, i+1])
        task_lists[i].append([i, i+2])
    print(task_lists)
    for agent_i_tasks in task_lists:
        if agent_i_tasks:
            for task in agent_i_tasks:
                
                agent_i, agent_A_t = task
                print(agent_i, agent_A_t)
                
                
                [[0] * num_task for _ in range(agent_num)]