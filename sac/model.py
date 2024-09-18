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
seq_length = 50  # LSTM序列长度
num_layers = 1  # LSTM层数


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
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
    
class LSTMCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(LSTMCritic, self).__init__()
        
        # Q1 architecture
        self.lstm = nn.LSTM(num_inputs, hidden_dim)
        self.fc_action = nn.Linear(num_actions, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.lstm = nn.LSTM(num_inputs, hidden_dim)
        self.fc_action = nn.Linear(num_actions, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim*2, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state, action, hidden=None):
        lstm_out, _ = self.lstm(state, hidden)
        action_emb = self.fc_action(action)
        out = torch.cat((lstm_out[:, -1, :], action_emb), dim=1)
        
        x1 = F.relu(self.linear1(out))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(out))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
    
# ----------------------------------------------------------------actor网络--------------------------------------------------------------
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
 
        self.mean_linear = nn.Linear(hidden_dim, num_actions) # 计算动作均值的线性层
        self.log_std_linear = nn.Linear(hidden_dim, num_actions) # 计算动作标准差对数的线性层

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

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # 限制在规定范围内
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std) # 使用预测的均值和标准差创建一个正态分布
        x_t = normal.rsample()  # 使用重参数化技巧从正态分布中采样，以确保梯度可以通过采样过程 (mean + std * N(0,1))
        y_t = torch.tanh(x_t) # 将其范围约束在[-1, 1]之间
        action = y_t * self.action_scale + self.action_bias
        
        log_prob = normal.log_prob(x_t)
        # 因为使用了 tanh 函数，所以需要修正日志概率，以反映动作被限制在实际动作空间边界内的事实
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        if state.dim()==2:
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = log_prob.sum(0, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

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

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
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
        return super(DeterministicPolicy, self).to(device)

# 定义LSTMActor类
class LSTMActorGaussian(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(LSTMActorGaussian, self).__init__()
        
        self.lstm = nn.LSTM(num_inputs, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, num_actions) # 计算动作均值的线性层
        self.log_std_linear = nn.Linear(hidden_dim, num_actions) # 计算动作标准差对数的线性层
        
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
            
        self.apply(weights_init_)
        
    def init_hidden(self, hidden_dim, device):
        # 初始化隐藏状态 h_0 和细胞状态 c_0
        h_0 = torch.zeros(num_layers, 10, hidden_dim).to(device)
        c_0 = torch.zeros(num_layers, 10, hidden_dim).to(device)
        return (h_0, c_0)
    
    def forward(self, state_sequence, h_c=None):
        lstm_out, h_c = self.lstm(state_sequence, h_c)
        lstm_out = lstm_out[:, -1, :] # 取最后一个时间步的输出
        x = F.relu(self.fc(lstm_out))  
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # 限制在规定范围内
        return mean, log_std, h_c
    
    def sample(self, state_sequence, h_c=None):
        mean, log_std, h_c = self.forward(state_sequence, h_c)
        std = log_std.exp()
        normal = Normal(mean, std) # 使用预测的均值和标准差创建一个正态分布
        x_t = normal.rsample()  # 使用重参数化技巧从正态分布中采样，以确保梯度可以通过采样过程 (mean + std * N(0,1))
        y_t = torch.tanh(x_t) # 将其范围约束在[-1, 1]之间
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # 因为使用了 tanh 函数，所以需要修正日志概率，以反映动作被限制在实际动作空间边界内的事实
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, h_c

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(LSTMActorGaussian, self).to(device)
    
class LSTMActorDeterministic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(LSTMActorDeterministic, self).__init__()

# 注意力层
class AttentionLayer(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.W_query = nn.Linear(feature_dim, hidden_dim)
        self.W_key = nn.Linear(feature_dim, hidden_dim)
        self.W_value = nn.Linear(feature_dim, hidden_dim)
        self.sqrt_scalar = np.sqrt(hidden_dim)

    def forward(self, query, keys):
        # query shape: (batch_size, feature_dim)
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
