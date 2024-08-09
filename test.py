import multiprocessing
import os
import sys
import time
import csv
import numpy as np
import torch
from gym import spaces

def index2ud(index, server_num, ud_num):
    server = index // ud_num
    ud = index - server * ud_num
    return server, ud

if __name__ == '__main__':
    directory = 'runs/data'
    filename = 'output.csv'
    file_path = os.path.join(directory, filename)
    data=[["a","b","c"],[1,2,3]]
    print(file_path)
    # 写入 CSV 文件
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # 写入表头
        writer.writerow(data[0])
        
        # 写入数据行
        for row in data[1:]:
            writer.writerow(row)

    print(f"Data has been written to {file_path}")