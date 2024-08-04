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
    last_use = [[0] * 8 for _ in range(6)]
    print(last_use)