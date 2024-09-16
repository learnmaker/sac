import torch
import random
from torch.utils.tensorboard import SummaryWriter
 
writer = SummaryWriter("logs")        #当前目录下创建logs文件夹用于存放日志
for i in range(100):
    writer.add_scalar("y = x*X", i*i+100, i)
writer.close()