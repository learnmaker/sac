import multiprocessing
import time
import numpy as np
import torch

def index2ud(index, server_num, ud_num):
    server = index // ud_num
    ud = index - server * ud_num
    return server, ud

if __name__ == '__main__':
    tensor_state = torch.FloatTensor(range(8))
    servers_cache_states = torch.FloatTensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(servers_cache_states.view(-1))
    print(tensor_state)
    print(torch.cat((servers_cache_states.view(-1), tensor_state),dim=0))