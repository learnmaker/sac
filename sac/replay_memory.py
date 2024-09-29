import os
import pickle
import random
import numpy as np
import torch

# 将 hc 转换回 PyTorch 张量
def to_tensor(hc, device):
    hc = torch.tensor(hc, dtype=torch.float32).to(device)
    # hc 的形状是 (batch_size, 2, num_layers, batch_size_per_layer, hidden_size)
    batch_size, _, num_layers, num_directions, hidden_size = hc.shape
    
    # 我们需要将其拆分为 h 和 c
    h_0 = hc[:, 0, ...]  # 获取所有的 h
    c_0 = hc[:, 1, ...]  # 获取所有的 c
    
    # 调整 h_0 和 c_0 的形状以符合 LSTM 层的预期
    h_0 = h_0.permute(1, 2, 0, 3).contiguous().view(num_layers * num_directions, batch_size, hidden_size)
    c_0 = c_0.permute(1, 2, 0, 3).contiguous().view(num_layers * num_directions, batch_size, hidden_size)
    
    return (h_0, c_0)

class ReplayMemory:
    def __init__(self, capacity, seed, device):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = device

    def push(self, state, action, reward, next_state, done, state_seq=None, next_state_seq=None, old_hc=None, hc=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        if state_seq is None:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, state_seq, next_state, old_hc, hc)
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, mold, batch_size):
        batch = random.sample(self.buffer, batch_size)
        if mold == 3:
            state, action, reward, next_state, done, state_seq, next_state, old_hc, hc= map(np.stack, zip(*batch))
            old_hc = to_tensor(old_hc, self.device)
            hc = to_tensor(hc, self.device)
            return state, action, reward, next_state, done, state_seq, next_state, old_hc, hc
        else:
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
