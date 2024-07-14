import multiprocessing
import time
import numpy as np

def index2ud(index, server_num, ud_num):
    server = index // ud_num
    ud = index - server * ud_num
    return server, ud

if __name__ == '__main__':
    print(index2ud(4,2,3))