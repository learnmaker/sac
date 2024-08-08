import multiprocessing
import sys
import time
import numpy as np
import torch
from gym import spaces

def index2ud(index, server_num, ud_num):
    server = index // ud_num
    ud = index - server * ud_num
    return server, ud

if __name__ == '__main__':
    # 获取C long类型的最大值
    c_long_max = sys.maxsize
    print("C long max size:", c_long_max)

    # 获取C long类型的实际位数
    c_long_bit_size = sys.maxsize.bit_length()
    print("C long bit size:", c_long_bit_size)