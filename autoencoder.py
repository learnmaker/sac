import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tool.data_loader import load_data

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    task_num = 8
    agent_num = 6
    global_info_num = agent_num + agent_num * task_num * 2

    # 准备全局信息数据
    global_info = load_data('./mydata/global_info/encode_data.csv')
    print("读取数据")

    # 转换为 PyTorch 的张量
    global_info_tensor = torch.tensor(global_info, dtype=torch.float32)
    dataset = TensorDataset(global_info_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 定义模型、损失函数和优化器
    input_dim = global_info_num  # 全局信息的维度
    hidden_dim = 32  # 降维后的维度
    model = Autoencoder(input_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练自编码器
    num_epochs = 100
    for epoch in range(num_epochs):
        for data in dataloader:
            inputs = data[0]
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, inputs)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存训练好的模型
    torch.save(model.state_dict(), 'autoencoder.pth')
    print("训练完成")

