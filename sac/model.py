import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from gym import spaces

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
seq_length = 5  # LSTM序列长度
num_layers = 2  # LSTM层数
lstm_hidden_size = 64 # LSTM隐藏层


# 遍历网络中的所有nn.Linear模块，并对它们的权重和偏置进行初始化
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1) # 使用Xavier均匀分布初始化权重矩阵
        torch.nn.init.constant_(m.bias, 0) # 所有的偏置项都被初始化为0

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_) # 在网络的所有子模块上调用weights_init_函数

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# ----------------------------------------------------------------critic网络--------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, local_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        # 处理state
        self.local_fc = nn.Linear(local_dim, hidden_dim)
        self.lstm = nn.LSTM(local_dim, hidden_dim)
        # 处理global_states
        self.global_attention = AttentionLayer(local_dim, lstm_hidden_size, num_layers, batch_first=True)
        # 处理action
        self.action_fc = nn.Linear(action_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(local_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc_info = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc_lstm = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(weights_init_)
        
    def forward(self, local_state, action):
        concatenated = torch.cat([local_state, action], dim=1)
        return self.fc(concatenated)
        
    def forward_info(self, local_state, global_states, action):
        # local_state + global_states ->hidden_dim
        attn_features = self.global_attention(local_state, global_states)
        # local_state -> hidden_dim
        local_state = self.local_fc(local_state)
        # action -> hidden_dim
        action = self.action_fc(action)
        concatenated = torch.cat((local_state, attn_features, action), dim=1)
        return self.fc_info(concatenated)
    
    def forward_lstm(self, local_state, hidden, global_states, action):
        lstm_out, _ = self.lstm(local_state, hidden)
        attn_features = self.global_attention(lstm_out, global_states)
        action = self.action_fc(action)
        concatenated = torch.cat((lstm_out[:, -1, :], attn_features, action), dim=1)
        return self.fc_lstm(concatenated)
    
# ----------------------------------------------------------------actor网络--------------------------------------------------------------
class GaussianActor(nn.Module):
    def __init__(self, local_dim, hidden_dim, action_space):
        super(GaussianActor, self).__init__()
        action_dim = action_space.shape[0]
        # 处理state
        self.local_fc = nn.Linear(local_dim, hidden_dim)
        self.lstm = nn.LSTM(local_dim, hidden_dim, num_layers)
        # 处理global_states
        self.global_attention = AttentionLayer(local_dim, hidden_dim)
        # 处理action
        self.action_fc = nn.Linear(action_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_info = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_lstm = nn.Sequential(
            nn.Linear(hidden_dim + local_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim) # 计算动作均值的线性层
        self.log_std_linear = nn.Linear(hidden_dim, action_dim) # 计算动作标准差对数的线性层

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
    
    def init_hidden(self, hidden_dim, device):
        # 初始化隐藏状态 h_0 和细胞状态 c_0
        h_0 = torch.zeros(num_layers, seq_length, hidden_dim).to(device)
        c_0 = torch.zeros(num_layers, seq_length, hidden_dim).to(device)
        return (h_0, c_0)
            
    def forward(self, local_state):
        x = self.fc(local_state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
    def forward_info(self, local_state, global_states):
        local_state = self.local_fc(local_state)
        attn_features = self.global_attention(local_state, global_states)
        concatenated = torch.cat((local_state, attn_features), dim=1)
        x = self.fc_info(concatenated)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # 限制在规定范围内
        return mean, log_std
    
    def forward_lstm(self, local_state, global_states, state_sequence, h_c):
        lstm_out, h_c = self.lstm(state_sequence, h_c)
        lstm_out = lstm_out[:, -1, :] # 取最后一个时间步的输出
        attn_features = self.global_attention(lstm_out, global_states)
        concatenated = torch.cat((local_state, attn_features), dim=1)
        x = self.fc_lstm(concatenated)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # 限制在规定范围内
        return mean, log_std, h_c

    def sample(self, mold, local_state, global_states=None, state_sequence=None, h_c=None):
        if mold == 1:
            mean, log_std = self.forward(local_state)
        elif mold == 2:
            mean, log_std = self.forward_info(local_state, global_states)
        else:
            mean, log_std, h_c = self.forward_lstm(local_state, global_states, state_sequence, h_c)
            
        std = log_std.exp()
        normal = Normal(mean, std) # 使用预测的均值和标准差创建一个正态分布
        x_t = normal.rsample()  # 使用重参数化技巧从正态分布中采样，以确保梯度可以通过采样过程 (mean + std * N(0,1))
        y_t = torch.tanh(x_t) # 将其范围约束在[-1, 1]之间
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # 因为使用了 tanh 函数，所以需要修正日志概率，以反映动作被限制在实际动作空间边界内的事实
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        if local_state.dim()==2:
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = log_prob.sum(0, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, h_c

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianActor, self).to(device)


class DeterministicActor(nn.Module):
    def __init__(self, local_dim, agent_num, hidden_dim, action_space):
        super(DeterministicActor, self).__init__()
        action_dim = action_space.shape[0]
        global_dim = local_dim * (agent_num-1)
        
        # 处理state
        self.local_fc = nn.Linear(local_dim, hidden_dim)
        self.lstm = nn.LSTM(local_dim, hidden_dim)
        # 处理global_states
        self.global_attention = AttentionLayer(global_dim, hidden_dim)
        # 处理action
        self.action_fc = nn.Linear(action_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_info = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_lstm = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.noise = torch.Tensor(action_dim)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, local_state):
        x = self.fc(local_state)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean
    
    def forward_info(self, local_state, global_states):
        local_state = self.local_fc(local_state)
        attn_features = self.global_attention(local_state, global_states)
        concatenated = torch.cat((local_state, attn_features), dim=1)
        x = self.fc_info(concatenated)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean
    
    def forward_lstm(self, local_state, global_states, hidden):
        lstm_out, _ = self.lstm(local_state, hidden)
        attn_features = self.global_attention(lstm_out, global_states)
        concatenated = torch.cat((local_state, attn_features), dim=1)
        x = self.fc_lstm(concatenated)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicActor, self).to(device)

# 注意力层
class AttentionLayer(nn.Module):
    def __init__(self, feature_dim, hidden_dim = 64):
        super(AttentionLayer, self).__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_key = nn.Linear(feature_dim, hidden_dim)
        self.W_value = nn.Linear(hidden_dim, hidden_dim)
        self.sqrt_scalar = np.sqrt(hidden_dim)

    def forward(self, query, keys):
        # query shape: (batch_size, hidden_dim)
        # keys shape: (batch_size, num_agents - 1, feature_dim)
        
        # Project queries and keys
        queries = self.W_query(query).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        keys = self.W_key(keys)  # (batch_size, num_agents - 1, hidden_dim)
        
        # Compute attention scores (scaled dot product attention)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / self.sqrt_scalar  # (batch_size, 1, num_agents - 1)
        attention_scores = F.softmax(attention_scores, dim=-1)  # Normalize
        
        # Apply attention to values
        values = self.W_value(keys)  # (batch_size, num_agents - 1, hidden_dim)
        weighted_values = torch.bmm(attention_scores, values)  # (batch_size, 1, hidden_dim)
        
        return weighted_values.squeeze(1)  # Remove batch dimension
