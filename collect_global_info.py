import argparse
import datetime
import math
import os
import numpy as np
import itertools
import torch
import sys
import csv
from autoencoder import Autoencoder
from sac.sac import SAC
from tool.generate_snrs import generate_snrs
from tool.samples_from_transmat import generate_request
from sac.replay_memory import ReplayMemory
from envs.MultiAgentEnv import MultiAgentEnv
from tool.data_loader import load_data
from config import system_config
from gym import spaces

def get_state_comb(state, server_requests, servers_cache_states):
    state = torch.FloatTensor(state)
    server_requests = torch.FloatTensor(server_requests)
    servers_cache_states = torch.FloatTensor(servers_cache_states)
    state_comb = torch.cat((state, server_requests, servers_cache_states.view(-1)), dim=0)
    return state_comb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='基于SAC算法的多服务器多用户设备的3C问题')
    # 环境名称
    parser.add_argument('--env-name', default="MultiAgentEnv",
                        help='环境名称 (default: MultiAgentEnv)')
    # 实验配置
    parser.add_argument('--exp-case', default="case3",
                        help='实验配置 (default: case 5)')
    # 策略类型
    parser.add_argument('--policy', default="Gaussian",
                        help='策略类型: Gaussian(正态) | Deterministic(确定) (default: Gaussian)')
    # 是否引入全局信息
    parser.add_argument('--global-info', action='store_true', default=False,
                        help='是否引入全局信息 (default: False)')
    # 是否使用LSTM
    parser.add_argument('--lstm', action='store_true', default=False,
                        help='是否使用LSTM (default: False)')
    # 每10次评估一次策略
    parser.add_argument('--eval', action='store_true', default=False,
                        help='是否评估 (default: False)')
    # 折现因子
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='reward折现因子 (default: 0.99)')
    # 目标平滑系数 θi_bar = ξ * θi + (1 - ξ) * θi_bar
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='目标平滑系数(τ) (default: 0.005)')
    # 学习率
    parser.add_argument('--lr', type=float, default=3e-4, metavar='G',
                        help='学习率 (default: 0.0003)')
    # 熵前面的温度系数
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='温度系数 (default: 0.2)')
    # 熵前面的系数是否自动调整
    parser.add_argument('--no_automatic_entropy_tuning', action='store_false', default=True,
                        help='α是否自动调整 (default: True)')
    # 随机种子
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='随机种子 (default: 123456)')
    # 批量大小
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='批量大小 (default: 256)')
    # 最大迭代次数
    parser.add_argument('--max_episode', type=int, default=300, metavar='N',
                        help='最大迭代次数 (default: 300)')
    # 隐藏层大小
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='隐藏层大小 (default: 256)')
    # 每次更新参数采样多少次
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='每次更新参数采样多少次 (default: 1)')
    # 使用策略网络决策动作前，多少次随机采样
    parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                        help='随机采样次数 (default: 10000)')
    # 目标网络的更新周期
    parser.add_argument('--target_update_interval', type=int, default=1000, metavar='N',
                        help='目标网络的更新周期 (default: 1000)')
    # 经验缓冲区大小
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='经验缓冲区大小 (default: 10000000)')
    # 是否使用cuda
    parser.add_argument('--cuda', action="store_true", default=False,
                        help='是否使用CUDA (default: False)')
    # 服务器个数
    parser.add_argument('--server_num', type=int, default=2, metavar='N',
                        help='服务器个数 (default: 2)')
    # 每个服务器的用户设备个数
    parser.add_argument('--ud_num', type=int, default=3, metavar='N',
                        help='每个服务器的用户设备个数 (default: 3)')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent_num = args.server_num * args.ud_num
    weight = system_config['weight']
    device = torch.device("cuda" if args.cuda else "cpu")
    encode_data = []
    
    task_num = system_config['F']  # 任务数 8
    maxp = system_config['maxp']   # 最大转移概率 70%
    task_utils = load_data('./mydata/task_info/task' + str(task_num) + '_utils.csv')  # 任务集信息[I, O, w，τ]
    task_set_ = task_utils.tolist()
    
    server_requests = np.full(agent_num, -1)
    servers_cache_states = np.full((agent_num, task_num, 2), 0)
    generate_snrs(args.server_num)  # 生成信噪比，每个服务器下的信噪比相同
    generate_request(args.server_num, args.ud_num, task_num, maxp)  # 生成任务请求
    
    snrs=[]
    Ats=[]
    agents = []
    
    # 动作空间
    sample_low = np.asarray([-1] * (3 * task_num + 2), dtype=np.float32)
    sample_high = np.asarray([1] * (3 * task_num + 2), dtype=np.float32)
    action_space = spaces.Box(low=sample_low, high=sample_high, dtype=np.float32)
    
    if args.global_info:
        state_dim = 2*task_num+1 + server_requests.size + servers_cache_states.size
    else:
        state_dim = 2*task_num+1
    
    for server in range(args.server_num):
        # 该服务器的信噪比
        snr = load_data('./mydata/temp/dynamic_snrs_' + str(server+1)+'.csv').reshape(1, -1)[0]
        for ud in range(args.ud_num):
            # 该用户设备的任务请求
            At = load_data("./mydata/temp/server"+str(server+1)+"_ud"+str(ud+1) + "_samples"+str(task_num)+"_maxp"+str(maxp)+".csv").reshape(1, -1)[0]
            Ats.append(At)
            snrs.append(snr)
            # 该用户设备的SAC网络
            agent = SAC(state_dim, action_space, args)
            agents.append(agent)
        
    # 系统状态[S^I, S^O, A(0)]、任务信息、任务请求、信噪比、策略类型       
    env = MultiAgentEnv(init_sys_states=[[0] * (2 * task_num) + [1] for _ in range(agent_num)], agent_num=agent_num, task_set=task_set_, requests=Ats,
                                channel_snrs=snrs, exp_case=args.exp_case)
    env.action_space.seed(args.seed)

    # 经验缓存区
    memories = [ReplayMemory(args.replay_size, args.seed) for _ in range(agent_num)]
    
    for i_episode in itertools.count(1):   # <------------------------------------ 回合数
        # 初始化
        episode_rewards = np.full(agent_num, 0.)
        episode_step = 0
        dones = np.full(agent_num, False) # 本回合各agent是否结束
        states = env.reset()
        
        server_requests = env.get_requests()
        servers_cache_states = env.get_cach_state()
            
        # 如果还有agent没有结束
        while np.sum(dones == False) > 0:   # <----------------------------------- 训练步数step
            
            actions=[]
            masks=[]
            
            encode_data.append(np.hstack((np.array(server_requests), np.array(servers_cache_states).flatten())))
                
            # 对每个agent进行训练
            for index in range(agent_num):
                
                done = dones[index]
                # done为True，跳过
                if done:
                    continue
                
                agent = agents[index]
                state = states[index]
                
                action = env.action_space.sample()  # 随机动作
                actions.append(action)
                
                mask = 1 if episode_step == env._max_episode_steps else float(not done)
                masks.append(mask)
                    
            next_states, rewards, new_dones, infos = env.step(actions)  # Step
            # print("当前step",episode_step)
            for i in range(agent_num):
                state_comb = get_state_comb(states[i], server_requests, servers_cache_states)
                server_requests = torch.FloatTensor(env.get_requests())
                servers_cache_states = torch.FloatTensor(env.get_cach_state())
                next_state_comb = get_state_comb(next_states[i], server_requests, servers_cache_states)
                memories[i].push(state_comb, actions[i], rewards[i], next_state_comb, masks[i])
                episode_rewards[i] += rewards[i]
                
            episode_step += 1
            states = next_states
            dones = new_dones
           
        if i_episode > args.max_episode:
            break
        print("进度：",i_episode,"/",args.max_episode, "本次写入",len(encode_data))
        
        file_path = os.path.join("mydata/global_info", "encode_data.csv")
        if i_episode==1:
            
            if not os.path.exists("mydata/global_info"):
                os.makedirs("mydata/global_info")
                
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer_encode = csv.writer(file)
                for row in encode_data:
                    writer_encode.writerow(row)
        else:
            with open(file_path, mode='a', newline='', encoding='utf-8') as file:
                writer_encode = csv.writer(file)
                for row in encode_data:
                    writer_encode.writerow(row)
        encode_data = []