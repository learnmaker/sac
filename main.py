import argparse
import datetime
import math
import os
import numpy as np
import itertools
import torch
import sys
import csv
from sac.sac import SAC
from tool.generate_snrs import generate_snrs
from tool.samples_from_transmat import generate_request
from torch.utils.tensorboard import SummaryWriter
from sac.replay_memory import ReplayMemory
from envs.MultiAgentEnv import MultiAgentEnv
from tool.data_loader import load_data
from config import system_config
from gym import spaces

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# agent编号转服务器、用户设备编号
def index2ud(index, ud_num):
    server = index // ud_num
    ud = index - server * ud_num
    return server, ud

# 获取最大位数
def find_max_digit_position(num):
    if num == 0:
        return 1  # 特殊情况：0 的最大位数为 1
    max_digit_position = int(math.log10(abs(num))) + 1
    return max_digit_position

# 设置实验数据保存名称
def set_fieldnames(agent_num):
    fd1=[]
    fd2=[]
    fd3=['avg_reward/test_episode', 'avg_cost/trans_cost', 'avg_cost/comp_cost', 'total_cost']
    for index in range(agent_num):
        server_index, ud_index = index2ud(index, args.ud_num)
        
        fd1.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'_loss/critic_1')
        fd1.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'_loss/critic_2')
        fd1.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'_loss/policy')
        fd1.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'_loss/entropy_loss')
        fd1.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'/alpha')
        
        fd2.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'_reward/train')
    fd2.append("total_reward")
    
    data1.append(fd1)
    data2.append(fd2)
    data3.append(fd3)
    return

# 添加时间序列
def add_state_sequence(i, states):
    local_state = np.array(local_state)
    global_state = np.array(global_state)
    
    if len(state_sequence[i]) < sequence_length:
        state_sequence[i].append(states)
    else:
        state_sequence[i] = state_sequence[i][1:]
        state_sequence[i].append(states)
    return

# 获取时间序列
def get_state_sequence(i, state_dim):
    current_length = len(state_sequence[i])
    # 如果当前时间步少于序列长度，用0填充
    if current_length < sequence_length:
        padding = [np.zeros(state_dim) for _ in range(sequence_length - current_length)]
        state_sequence_new = padding + state_sequence[i]
    else:
        state_sequence_new = state_sequence[i]
    state_sequence_np = np.array(state_sequence_new)
    state_sequence_tensor = torch.from_numpy(state_sequence_np).float()
    return  state_sequence_tensor

# 展示系统状态
def show_states(states):
    for index in range(len(states)):
        state = states[index]
        print("agent",index,"state:",state)
        
# 保存实验数据位置
data_directory = "runs/"
filename1 = "update_parameters.csv"
data1 = []
filename2 = "episode_rewards.csv"
data2 = []
filename3 = "eval.csv"
data3 = []

# LSTM序列长度
sequence_length = 10
state_sequence = []

if __name__ == '__main__':
    # ------------------------------------------------------1. 命令行参数设置-----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='基于SAC算法的多服务器多用户设备的3C问题')
    # 环境名称
    parser.add_argument('--env-name', default="MultiAgentEnv",
                        help='环境名称 (default: MultiAgentEnv)')
    # 实验配置
    parser.add_argument('--exp-case', default="case5",
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
    # 是否每10次评估一次策略
    parser.add_argument('--eval', action='store_true', default=False,
                        help='是否评估 (default: False)')
    # 折现因子
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='reward折现因子 (default: 0.99)')
    # 目标平滑系数 θi_bar = ξ * θi + (1 - ξ) * θi_bar
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='目标平滑系数(τ) (default: 0.005)')
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-4, metavar='G',
                        help='学习率 (default: 0.0004)')
    # 熵前面的温度系数
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='温度系数 (default: 0.2)')
    # 熵前面的系数是否自动调整
    parser.add_argument('--automatic_entropy_tuning', action='store_true', default=False,
                        help='α是否自动调整 (default: False)')
    # 随机种子
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='随机种子 (default: 123456)')
    # 批量大小
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='批量大小 (default: 512)')
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

    # ------------------------------------------------------2. 环境设置-----------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    agent_num = args.server_num * args.ud_num
    weight = system_config['weight'] # weight = 1
    device = torch.device("cuda" if args.cuda else "cpu")
    state_sequence = [[] for _ in range(agent_num)]
    task_num = system_config['F']  # 任务数 8
    maxp = system_config['maxp']   # 最大转移概率 70%
    task_utils = load_data('./mydata/task_info/task' + str(task_num) + '_utils.csv')  # 任务集信息[I, O, w，τ]
    task_set_ = task_utils.tolist()
    
    if args.lstm:
        mold = 3
    elif args.global_info:
        mold = 2
    else:
        mold = 1
    
    # 设置表头
    set_fieldnames(agent_num)
    
    if args.global_info:
        # 保存所有用户设备的任务请求，agent_num，初始化为-1
        server_requests = np.full(agent_num, -1)
        # 保存所有用户设备的缓存状态，agent_num * task_num * 2，初始化为0
        servers_cache_states = np.full((agent_num, task_num, 2), 0)

    # 跟据服务器数量和用户设备数量生成 任务请求和信噪比，保存在temp文件夹
    generate_snrs(args.server_num)  # 生成信噪比，每个服务器下的信噪比相同
    generate_request(args.server_num, args.ud_num, task_num, maxp)  # 生成任务请求

    # Tensorboard保存实验数据
    filename = '{}_SAC_{}{}{}{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                args.exp_case, 
                                "_global-info" if args.global_info else "",
                                "_offload" if args.exp_case == "case5" else "",
                                "_lstm" if args.lstm else "",
                                )
    writer = SummaryWriter('runs/' + filename)
    data_directory = data_directory + filename
    
    snrs=[]
    Ats=[]
    agents = []
    
    # 动作空间
    sample_low = np.asarray([-1] * (3 * task_num + 2), dtype=np.float32)
    sample_high = np.asarray([1] * (3 * task_num + 2), dtype=np.float32)
    action_space = spaces.Box(low=sample_low, high=sample_high, dtype=np.float32)
    local_dim = 2*task_num+1
    
    for server in range(args.server_num):
        # 该服务器的信噪比
        snr = load_data('./mydata/temp/dynamic_snrs_' + str(server+1)+'.csv').reshape(1, -1)[0]
        for ud in range(args.ud_num):
            # 该用户设备的任务请求
            At = load_data("./mydata/temp/server"+str(server+1)+"_ud"+str(ud+1) + "_samples"+str(task_num)+"_maxp"+str(maxp)+".csv").reshape(1, -1)[0]
            Ats.append(At)
            snrs.append(snr)
            # 该用户设备的SAC网络
            agent = SAC(local_dim, action_space, args)
            agents.append(agent)
        
    # 系统状态[S^I, S^O, A(0)]、任务信息、任务请求、信噪比、策略类型       
    env = MultiAgentEnv(init_sys_states=[[0] * (2 * task_num) + [1] for _ in range(agent_num)], agent_num=agent_num, task_set=task_set_, requests=Ats,
                                channel_snrs=snrs, exp_case=args.exp_case)
    env.action_space.seed(args.seed)

    # 经验缓存区
    memories = [ReplayMemory(args.replay_size, args.seed) for _ in range(agent_num)]
    
    # ------------------------------------------------------3. 训练-----------------------------------------------------------------------------
    print("环境初始化完毕，开始训练")
    total_numsteps = 0  # 总训练步数
    updates = 0  # 总更新参数次数
    result_trans = []  # 保存传输消耗评估结果
    result_comp = []  # 保存计算消耗评估结果
    
    for i_episode in itertools.count(1):   # <------------------------------------ 回合数
        # 初始化
        episode_rewards = np.full(agent_num, 0.)
        episode_step = 0
        dones = np.full(agent_num, False) # 本回合各agent是否结束
        states = env.reset()
        if args.lstm:
            h_cs = [agent.actor.init_hidden(args.hidden_size, device) for agent in agents]
            
        # 如果还有agent没有结束
        while np.sum(dones == False) > 0:   # <----------------------------------- 训练步数step
            
            actions=[]
            masks=[]
            temp_data1=[]
                
            # 对每个agent进行训练
            for index in range(agent_num):
                
                done = dones[index]
                # done为True，跳过
                if done:
                    continue
                
                agent = agents[index]
                state = states[index]
                
                if args.start_steps > total_numsteps:
                    action = env.action_space.sample()  # 随机动作
                else:
                    if args.lstm:
                        state_seq = get_state_sequence(index, local_dim)
                        action, h_cs[index] = agent.select_action_lstm(state_seq, h_cs[index])
                    else:
                        if args.global_info:
                            action = agent.select_action_info(index, states)
                        else:
                            action = agent.select_action(state)

                server_index, ud_index = index2ud(index, args.ud_num)
                actions.append(action)
                
                mask = 1 if episode_step == env._max_episode_steps else float(not done)
                masks.append(mask)

                if len(memories[index]) > args.batch_size:
                    # Number of updates per step in environment
                    for i in range(args.updates_per_step):
                        # Update parameters of all the networks
                        critic_loss, actor_loss, ent_loss, alpha = agent.update_parameters(
                            index, memories[index], args.batch_size, updates, mold)
                        writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(
                            ud_index+1)+'_loss/critic', critic_loss, updates)
                        writer.add_scalar('server'+str(server_index+1)+'_userDevice' +
                                        str(ud_index+1)+'loss/actor', actor_loss, updates)
                        writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(
                            ud_index+1)+'loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(
                            ud_index+1)+'entropy_temprature/alpha', alpha, updates)
                        temp_data1.append(critic_loss)
                        temp_data1.append(actor_loss)
                        temp_data1.append(ent_loss)
                        temp_data1.append(alpha)
                        updates += 1
            
            if temp_data1:
                data1.append(temp_data1)
                    
            next_states, rewards, new_dones, infos = env.step(actions)  # Step
            
            # 经验缓存
            for i in range(agent_num):
                if args.lstm:
                    add_state_sequence(i, states)
                    next_state_seq = get_state_sequence(i, local_dim)
                    memories[i].push(state_seq, actions[i], rewards[i], next_state_seq, masks[i])
                else:
                    if args.global_info:
                        memories[i].push(states, actions[i], rewards[i], next_states, masks[i])
                    else:
                        memories[i].push(states[i], actions[i], rewards[i], next_states[i], masks[i])
                    
                episode_rewards[i] += rewards[i]
                
            episode_step += 1
            states = next_states
            dones = new_dones
            total_numsteps += 1
           
        if i_episode > args.max_episode:
            break
        
        print("Episode: {}, 总训练步数: {}, 本回合步数: {}, 平均回报: {}, 总回报：{}, 总回报最大位: 10**{}".format(i_episode, total_numsteps, episode_step, np.sum(episode_rewards)/agent_num, np.sum(episode_rewards), find_max_digit_position(np.sum(episode_rewards))))

        temp_data2 = []
        for index in range(agent_num):
            server_index, ud_index = index2ud(index, args.ud_num)
            writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'reward/train', episode_rewards[index], i_episode)
            temp_data2.append(episode_rewards[index])
            print("server{}_userDevice{}_reward: {}".format(server_index + 1, ud_index + 1, round(episode_rewards[index], 2)))
        temp_data2.append(sum(temp_data2))
        data2.append(temp_data2)
        
        # 评估
        eval_freq = 5  # 评估频率
        if i_episode % eval_freq == 0 and args.eval:
            
            avg_reward = 0
            avg_trans_cost = 0
            avg_compute_cost = 0
            episodes = 5  # 取5次的平均值，计算网络的奖励
            done_step = 0
            
            for _ in range(episodes):
                episode_reward = 0
                trans_cost = 0
                compute_cost = 0
                states = env.reset()
                dones = np.full(agent_num, False)
                if args.global_info:
                    if args.lstm:
                        h_cs = [agent.actor.init_hidden(args.hidden_size, device) for agent in agents]
                        
                while np.sum(dones == False) > 0:
                    
                    actions=[]
                    
                    for index in range(agent_num):
                        
                        done = dones[index]
                        if done:
                            continue
                        
                        agent = agents[index]
                        state = states[index]
                        if args.lstm:
                            state_seq = get_state_sequence(index, local_dim)
                            action, h_cs[index] = agent.select_action_lstm(state_seq, h_cs[index])
                        else:
                            if args.global_info:
                                action = agent.select_action_info(index, states)
                            else:
                                action = agent.select_action(state)
                        actions.append(action)
                        
                    next_states, rewards, new_dones, infos = env.step(actions)
                    if args.lstm:
                        for i in range(agent_num):
                            add_state_sequence(i, states)
                            
                    episode_reward += np.sum(rewards)
                        
                    try:
                        trans_cost += np.sum(infos['observe_details'][0])
                        compute_cost += np.sum(infos['observe_details'][1])
                        done_step += 1
                    except:
                        pass
                    
                    states = next_states
                    dones = new_dones

                avg_reward += episode_reward
                avg_trans_cost += trans_cost
                avg_compute_cost += compute_cost
            
            # 所有agent的总体reward、trans_cost、compute_cost
            avg_reward /= episodes
            avg_trans_cost /= done_step
            avg_compute_cost /= done_step
            
            writer.add_scalar('avg_reward/test_episode', avg_reward, i_episode)
            print("----------------------------------------")
            print("Test Episodes: {}, Total Steps: {}, Avg. Reward: {}, Avg. Trans Cost: {}, Avg. Compute Cost: {}, Glb. Step: {}".format(
                episodes, int(done_step), round(avg_reward, 2), round(avg_trans_cost, 2), round(avg_compute_cost, 2), env.global_step))
            print("----------------------------------------")
            result_trans.append(avg_trans_cost)
            result_comp.append(avg_compute_cost)

            if len(result_trans) > 10:
                print_avg_trans = np.average(np.asarray(result_trans[-10:]))
                print_avg_comp = np.average(np.asarray(result_comp[-10:]))
            else:
                print_avg_trans = np.average(np.asarray(result_trans))
                print_avg_comp = np.average(np.asarray(result_comp))
            print("Final Avg Results for last 100 epoches: Avg. Trans Cost: {}, Avg. Compute Cost: {}".format(
                round(print_avg_trans, 2), round(print_avg_comp, 2)))
            print("----------------------------------------")
            writer.add_scalar('avg_cost/trans_cost', round(print_avg_trans, 2), i_episode)
            writer.add_scalar('avg_cost/comp_cost', round(print_avg_comp, 2), i_episode)
            data3.append([avg_reward, round(print_avg_trans, 2), round(print_avg_comp, 2), round(print_avg_trans, 2) + weight*round(print_avg_comp, 2)])

        # -------------------------------------------------每回合结束写入一次数据-----------------------------------------------------
        file_path1=os.path.join(data_directory, filename1)
        file_path2=os.path.join(data_directory, filename2)
        file_path3=os.path.join(data_directory, filename3)
        
        if i_episode==1:
            if not os.path.exists(data_directory):
                os.makedirs(data_directory)
            # 写入表头
            with open(file_path1, mode='w', newline='', encoding='utf-8') as file:
                writer1 = csv.writer(file)
                for row in data1:
                    writer1.writerow(row)
            with open(file_path2, mode='w', newline='', encoding='utf-8') as file:
                writer2 = csv.writer(file)
                for row in data2:
                    writer2.writerow(row)
            with open(file_path3, mode='w', newline='', encoding='utf-8') as file:
                writer3 = csv.writer(file)
                for row in data3:
                    writer3.writerow(row)
        else:
            with open(file_path1, mode='a', newline='', encoding='utf-8') as file:
                writer1 = csv.writer(file)
                for row in data1:
                    writer1.writerow(row)
            with open(file_path2, mode='a', newline='', encoding='utf-8') as file:
                writer2 = csv.writer(file)
                for row in data2:
                    writer2.writerow(row)
            with open(file_path3, mode='a', newline='', encoding='utf-8') as file:
                writer3 = csv.writer(file)
                for row in data3:
                    writer3.writerow(row)
        data1 = []
        data2 = []
        data3 = []
    print("不合格动作：",env.getNotValid())