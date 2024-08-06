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
    list1 = [1, 2, 3, 4]
    list2 = [5, 6, 7, 8]

    # 转换为 NumPy 数组
    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # 或者直接使用 *
    result_direct = arr1 * arr2 + 5 + arr1
    print(result_direct)  # 输出: array([ 5, 12, 21, 32])