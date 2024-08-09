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
from envs.MultiTaskCore import MultiTaskCore
from envs.MultiAgentEnv import MultiAgentEnv
from tool.data_loader import load_data
from config import system_config
from gym import spaces

def index2ud(index, ud_num):
    server = index // ud_num
    ud = index - server * ud_num
    return server, ud

def find_max_digit_position(num):
    if num == 0:
        return 1  # 特殊情况：0 的最大位数为 1
    max_digit_position = int(math.log10(abs(num))) + 1
    return max_digit_position
def set_fieldnames(agent_num):
    fd1=[]
    fd2=[]
    fd3=['avg_reward/test_episode', 'avg_cost/trans_cost', 'avg_cost/comp_cost']
    for index in range(agent_num):
        server_index, ud_index = index2ud(index, args.ud_num)
        fd1.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'_loss/critic_1')
        fd1.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'loss/critic_2')
        fd1.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'loss/policy')
        fd1.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'loss/entropy_loss')
        fd1.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'entropy_temprature/alpha')
        fd2.append('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'reward/train')
    fd2.append("total_reward")
    data1.append(fd1)
    data2.append(fd2)
    data3.append(fd3)
    return
    
        
data_directory = "runs/data"
filename1 = "update_parameters.csv"
data1 = []
filename2 = "episode_rewards.csv"
data2 = []
filename3 = "eval.csv"
data3 = []

if __name__ == '__main__':
    # ------------------------------------------------------1. 命令行参数设置-----------------------------------------------------------------------
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
    # 每10次评估一次策略
    parser.add_argument('--eval', type=bool, default=True,
                        help='每 10 episode评估一次策略 (default: True)')
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
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='α是否自动调整 (default: True)')
    # 随机种子
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='随机种子 (default: 123456)')
    # 批量大小
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='批量大小 (default: 256)')
    # 最大训练步数
    parser.add_argument('--num_steps', type=int, default=5000001, metavar='N',
                        help='最大训练步数 (default: 5000001)')
    # 隐藏层大小
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='隐藏层大小 (default: 256)')
    # 每次更新参数采样多少次
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='每次更新参数采样多少次 (default: 1)')
    # 使用策略网络决策动作前，多少次随机采样
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='随机采样次数 (default: 10000)')
    # 目标网络的更新周期
    parser.add_argument('--target_update_interval', type=int, default=1000, metavar='N',
                        help='目标网络的更新周期 (default: 1000)')
    # 经验缓冲区大小
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='经验缓冲区大小 (default: 10000000)')
    # 是否使用cuda
    parser.add_argument('--cuda', action="store_true", default=True,
                        help='是否使用CUDA (default: True)')
    # 服务器个数
    parser.add_argument('--server_num', type=int, default=2,
                        help='服务器个数 (default: 2)')
    # 每个服务器的用户设备个数
    parser.add_argument('--ud_num', type=int, default=3,
                        help='每个服务器的用户设备个数 (default: 3)')
    args = parser.parse_args()

    # ------------------------------------------------------2. 环境设置-----------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent_num = args.server_num * args.ud_num
    set_fieldnames(agent_num)
    
    # 保存所有用户设备的任务请求，agent_num，初始化为-1
    server_requests = np.full(agent_num, -1)

    # 保存所有用户设备的缓存状态，agent_num * task_num * 2，初始化为0
    task_num = system_config['F']  # 任务数 8
    maxp = system_config['maxp']   # 最大转移概率 70%
    task_utils = load_data('./mydata/task_info/task' + str(task_num) + '_utils.csv')  # 任务集信息[I, O, w，τ]
    task_set_ = task_utils.tolist()
    servers_cache_states = np.full((agent_num, task_num, 2), 0)

    # 跟据服务器数量和用户设备数量生成 任务请求和信噪比，保存在temp文件夹
    generate_snrs(args.server_num)  # 生成信噪比，每个服务器下的信噪比相同
    generate_request(args.server_num, args.ud_num, task_num, maxp)  # 生成任务请求

    # Tensorboard保存实验数据
    writer = SummaryWriter(
        'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.exp_case,
                                      args.policy, "autotune" if args.automatic_entropy_tuning else ""))
    filename1 = '{}_SAC_{}_{}_{}_'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.exp_case,
                                      args.policy, "autotune" if args.automatic_entropy_tuning else "") + filename1
    filename2 = '{}_SAC_{}_{}_{}_'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.exp_case,
                                      args.policy, "autotune" if args.automatic_entropy_tuning else "") + filename2
    filename3 = '{}_SAC_{}_{}_{}_'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.exp_case,
                                      args.policy, "autotune" if args.automatic_entropy_tuning else "") + filename3
    snrs=[]
    Ats=[]
    agents = []
    
    # 动作空间
    sample_low = np.asarray([-1] * (3 * task_num + 2), dtype=np.float32)
    sample_high = np.asarray([1] * (3 * task_num + 2), dtype=np.float32)
    action_space = spaces.Box(low=sample_low, high=sample_high, dtype=np.float32)

    for server in range(args.server_num):
        # 该服务器的信噪比
        snr = load_data('./mydata/temp/dynamic_snrs_' + str(server+1)+'.csv').reshape(1, -1)[0]
        for ud in range(args.ud_num):
            # 该用户设备的任务请求
            At = load_data("./mydata/temp/server"+str(server+1)+"_ud"+str(ud+1) + "_samples"+str(task_num)+"_maxp"+str(maxp)+".csv").reshape(1, -1)[0]
            Ats.append(At)
            snrs.append(snr)
            # 该用户设备的SAC网络
            agent = SAC(2*task_num+1 + server_requests.size + servers_cache_states.size, action_space, args)
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
                    action = agent.select_action(state, server_requests, servers_cache_states)
                    
                server_index, ud_index = index2ud(index, args.ud_num)
                actions.append(action)
                
                mask = 1 if episode_step == env._max_episode_steps else float(not done)
                masks.append(mask)

                if len(memories[index]) > args.batch_size:
                    temp_data1=[]
                    # Number of updates per step in environment
                    for i in range(args.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                            memories[index], args.batch_size, updates)
                        writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(
                            ud_index+1)+'_loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(
                            ud_index+1)+'loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('server'+str(server_index+1)+'_userDevice' +
                                          str(ud_index+1)+'loss/policy', policy_loss, updates)
                        writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(
                            ud_index+1)+'loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(
                            ud_index+1)+'entropy_temprature/alpha', alpha, updates)
                        temp_data1.append(critic_1_loss)
                        temp_data1.append(critic_2_loss)
                        temp_data1.append(policy_loss)
                        temp_data1.append(ent_loss)
                        temp_data1.append(alpha)
                        updates += 1
                    data1.append(temp_data1)   
                    
            next_states, rewards, new_dones, infos = env.step(actions)  # Step
            old_requests = server_requests
            old_cach = servers_cache_states
            server_requests = env.get_requests()
            servers_cache_states = env.get_cach_state()
            
            for agent_i in range(agent_num):
                memories[agent_i].push(states[agent_i], old_requests, old_cach, actions[agent_i], rewards[agent_i], next_states[agent_i], server_requests, servers_cache_states, masks[agent_i])
                episode_rewards[agent_i] += rewards[agent_i]
                
            episode_step += 1
            
            states = next_states
            dones = new_dones
            
            total_numsteps += 1
             
        if total_numsteps > args.num_steps:
            break

        print("Episode: {}, 总训练步数: {}, 本回合步数: {}, 总回报：{}, 最大位: 10**{}".format(i_episode, total_numsteps, episode_step, np.sum(episode_rewards), find_max_digit_position(np.sum(episode_rewards))))
        temp_data2 = []
        for index in range(agent_num):
            server_index, ud_index = index2ud(index, args.ud_num)
            writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'reward/train', episode_rewards[index], i_episode)
            temp_data2.append(episode_rewards[index])
            print("server{}_userDevice{}_reward: {}".format(server_index + 1, ud_index + 1, round(episode_rewards[index], 2)))
        data2.append(temp_data2)

        # 评估
        eval_freq = 10  # 评估频率
        if i_episode % eval_freq == 0 and args.eval is True:
            
            avg_reward = 0
            avg_trans_cost = 0
            avg_compute_cost = 0
            episodes = 10  # 取10次的平均值，计算网络的奖励
            done_step = 0
            
            for _ in range(episodes):
                episode_reward = 0
                trans_cost = 0
                compute_cost = 0
                states = env.reset()
                dones = np.full(agent_num, False)
                server_requests = env.get_requests()
                ervers_cache_states = env.get_cach_state()
                
                while np.sum(dones == False) > 0:
                    
                    actions=[]
                    
                    for index in range(agent_num):
                        
                        done = dones[index]
                        if done:
                            continue
                        
                        agent = agents[index]
                        state = states[index]
                        
                        action = agent.select_action(state, server_requests, servers_cache_states)
                        actions.append(action)
                        
                    next_states, rewards, new_dones, infos = env.step(actions)
                    # 执行动作后，立刻更新任务缓存
                    server_requests = env.get_requests()
                    ervers_cache_states = env.get_cach_state()
                    
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
            data3.append([avg_reward, round(print_avg_trans, 2), round(print_avg_comp, 2)])

        # 每回合结束写入一次数据
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