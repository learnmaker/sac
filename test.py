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
    action_space = spaces.Box(low=[0], high=[5], dtype=np.float32)
    action_space.np_random