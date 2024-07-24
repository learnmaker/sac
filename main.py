import argparse
import datetime
import numpy as np
import itertools
import torch
import sys
from sac.sac import SAC
from tool.generate_snrs import generate_snrs
from tool.samples_from_transmat import generate_request
from torch.utils.tensorboard import SummaryWriter
from sac.replay_memory import ReplayMemory
from envs.MultiTaskCore import MultiTaskCore
from tool.data_loader import load_data
from config import system_config

# 收集总体的任务请求、缓存情况
def collect_info(envs, task_num, dones):
    new_request = []
    new_cache = []

    for env, done in zip(envs, dones):
        # 已经结束的设置为-1
        if done:
            cache = [-1] * 2 * task_num
            request = -1
        else:
            cache = env.system_state()[:-1]
            request = env.system_state()[-1]

        reordered_array = np.empty_like(cache)
        reordered_array[::2] = cache[:task_num]
        reordered_array[1::2] = cache[task_num:]
        cache = reordered_array.reshape(task_num, 2)

        new_cache.append(cache)
        new_request.append(request)
    return np.array(new_request), np.array(new_cache)


def index2ud(index, ud_num):
    server = index // ud_num
    ud = index - server * ud_num
    return server, ud


if __name__ == '__main__':
    # 命令行参数设置
    parser = argparse.ArgumentParser(description='SAC算法参数')
    # 环境名称
    parser.add_argument('--env-name', default="MultiTaskCore",
                        help='Wireless Comm environment (default: MultiTaskCore)')
    # 实验配置
    parser.add_argument('--exp-case', default="case3",
                        help='The experiment configuration case (default: case 3)')
    # 策略类型
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    # 每10次评估一次策略
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    # 折现因子
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    # 目标平滑系数 θi_bar = ξ * θi + (1 - ξ) * θi_bar
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-4, metavar='G',
                        help='learning rate (default: 0.0003)')
    # 熵前面的温度系数
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    # 熵前面的系数是否自动调整
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    # 随机种子
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    # 抽样大小
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    # 最大训练步数
    parser.add_argument('--num_steps', type=int, default=5000001, metavar='N',
                        help='maximum number of steps (default: 5000001)')
    # 隐藏层大小
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    # 每次更新参数采样多少次
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    # 使用策略网络决策动作前，多少次随机采样
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling1 random actions (default: 10000)')
    # 目标网络的更新周期
    parser.add_argument('--target_update_interval', type=int, default=1000, metavar='N',
                        help='Value target update per no. of updates per step (default: 1000)')
    # 经验缓冲区大小
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    # 是否使用cuda
    parser.add_argument('--cuda', action="store_true", default=True,
                        help='run on CUDA (default: False)')
    # 服务器个数
    parser.add_argument('--server_num', type=int, default=2,
                        help='server number(default: 2)')
    # 每个服务器的用户设备个数
    parser.add_argument('--ud_num', type=int, default=2,
                        help='user device number(default: 3)')
    args = parser.parse_args()

    # 环境设置
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent_num = args.server_num * args.ud_num

    # 保存所有用户设备的任务请求，1 * agent_num，初始化为-1
    server_requests = np.full(agent_num, -1)

    # 保存所有用户设备的缓存状态，agent_num * task_num * 2，初始化为0
    task_num = system_config['F']  # 任务数 6
    maxp = system_config['maxp']   # 最大转移概率 70%
    task_utils = load_data('./mydata/task_info/task' + str(task_num) + '_utils.csv')  # 任务集信息[I, O, w，τ]
    task_set_ = task_utils.tolist()
    servers_cache_states = np.full((agent_num, task_num, 2), 0)

    # 跟据服务器数量和用户设备数量生成 任务请求和信噪比，保存在temp文件夹
    generate_snrs(args.server_num)  # 生成信噪比，每个服务器下的信噪比相同
    generate_request(args.server_num, args.ud_num, task_num, maxp)  # 生成任务请求

    # Tensorboard保存实验数据
    writer = SummaryWriter(
        'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                      args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    envs = []
    agents = []
    memories = []
    for server in range(args.server_num):
        # 该服务器的信噪比
        snr = load_data('./mydata/temp/dynamic_snrs_' +
                        str(server+1)+'.csv').reshape(1, -1)[0]
        for ud in range(args.ud_num):
            # 该用户设备的任务请求
            At = load_data("./mydata/temp/server"+str(server+1)+"_ud"+str(ud+1) +
                           "_samples"+str(task_num)+"_maxp"+str(maxp)+".csv").reshape(1, -1)[0]
            # 系统状态[S^I, S^O, A(0)]、任务信息、任务请求、信噪比、策略类型
            # 单个用户设备
            env = MultiTaskCore(init_sys_state=[0] * (2 * task_num) + [1], agent_num=agent_num, task_set=task_set_, requests=At,
                                channel_snrs=snr, exp_case=args.exp_case)
            env.action_space.seed(args.seed)
            # 该用户设备的SAC网络
            agent = SAC(env.observation_space.shape[0], env.action_space, args)
            # 该用户设备的SAC网络的经验缓存区
            memory = ReplayMemory(args.replay_size, args.seed)

            envs.append(env)
            agents.append(agent)
            memories.append(memory)

    print("环境初始化完毕，开始训练")

    agent_numsteps = 0  # 总训练步数
    updates = 0  # 总更新参数次数
    result_trans = []  # 保存传输消耗评估结果
    result_comp = []  # 保存计算消耗评估结果

    for i_episode in itertools.count(1):  # 回合数
        episode_rewards = np.full(agent_num, 0) # 本回合各agent奖励
        episode_steps = np.full(agent_num, 0) # 本回合各agent交互步数
        dones = np.full(agent_num, False) # 本回合各agent是否结束
        states = [env.reset() for env in envs]

        # 如果还有agent没有结束
        while np.sum(dones == False) > 0:  # 训练步数
            
            # 每个agent上传自己的任务请求和缓存情况
            server_requests, servers_cache_states = collect_info(envs, task_num, dones)
            
            
            # 对每个agent进行训练
            for index, (env, agent, memory, done, state) in enumerate(zip(envs, agents, memories, dones, states)):

                # done为True，跳过
                if done:
                    continue

                if args.start_steps > agent_numsteps:
                    action = env.action_space.sample()  # 随机动作
                else:
                    action = agent.select_action(state, server_requests, servers_cache_states)  # 策略动作，传入请求和缓存
                    
                    
                server_index, ud_index = index2ud(index, args.ud_num)
                
                if len(memory) > args.batch_size:
                    # Number of updates per step in environment
                    for i in range(args.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                            memory, args.batch_size, updates)
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
                        updates += 1

                next_state, reward, done, info = env.step(action)  # Step
                episode_rewards[index] += reward
                episode_steps[index] += 1
                
                mask = 1 if episode_steps[index] == env._max_episode_steps else float(not done)
                memory.push(state, action, reward, next_state, mask)
                states[index] = next_state
                dones[index] = done
                
            agent_numsteps += 1

             
        if agent_numsteps > args.num_steps:
            break

        print("Episode: {}, total numsteps: {}, episode steps: {}".format(i_episode, agent_numsteps, episode_steps[index]))
        for index in range(agent_num):
            server_index, ud_index = index2ud(index, args.ud_num)
            writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(
                ud_index+1)+'reward/train', episode_rewards[index], i_episode)
            print("server{}_userDevice{}_reward: {}".format(server_index + 1, ud_index + 1, round(episode_rewards[index], 2)))


        # 评估
        eval_freq = 10  # 评估频率
        if i_episode % eval_freq == 0 and args.eval is True:
            
            avg_rewards = np.full(agent_num, 0)
            avg_trans_costs = np.full(agent_num, 0)
            avg_compute_costs = np.full(agent_num, 0)
            episodes = 10  # 取10次的平均值，计算网络的奖励
            done_steps = np.full(agent_num, 0)
            
            for _ in range(episodes):
                
                states = [env.reset() for env in envs]
                episode_rewards = np.full(agent_num, 0)
                trans_costs = np.full(agent_num, 0)
                compute_costs = np.full(agent_num, 0)
                dones = np.full(agent_num, False)
                
                while np.sum(dones == False) > 0:
                    for index, (env, agent, memory, done, state) in enumerate(zip(envs, agents, memories, dones, states)):
                        
                        if done:
                            continue
            
                        action = agent.select_action(state, evaluate=True)
                        next_state, reward, done, info = env.step(action)
                        episode_rewards[index] += reward
                        
                        try:
                            trans_costs[index] += info['observe_detail'][0]
                            compute_costs[index] += info['observe_detail'][1]
                            done_steps[index] += 1
                        except:
                            pass
                        states[index] = next_state
                        dones[index] = done

                for index in range(agent_num):
                    server_index, ud_index = index2ud(index, args.ud_num)
                    avg_rewards[index] += episode_rewards[index]
                    avg_trans_costs[index] += trans_costs[index]
                    avg_compute_costs[index] += compute_costs[index]

            print("----------------------------------------")
            for index in range(agent_num):
                server_index, ud_index = index2ud(index, args.ud_num)
                avg_rewards[index] /= episodes  # 每次训练的奖励
                avg_trans_costs[index] /= done_steps[index]  # 每步的传输消耗
                avg_compute_costs[index] /= done_steps[index]  # 每步的计算消耗
                writer.add_scalar('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+'_avgReward/test', avg_rewards[index], i_episode)
                print('server'+str(server_index+1)+'_userDevice'+str(ud_index+1)+"测试回合: {}, Total Steps: {}, Avg. Reward: {}, Avg. Trans Cost: {}, Avg. Compute Cost: {}, Glb. Step: {}".format(
                    episodes, int(done_steps[index]), round(avg_rewards[index], 2), round(avg_trans_costs[index], 2), round(avg_compute_costs[index], 2), env.global_step))
                result_trans.append(avg_trans_costs[index])
                result_comp.append(avg_compute_costs[index])
            print("----------------------------------------")
            
            if len(result_trans) > 10:
                print_avg_trans = np.average(
                    np.asarray(result_trans[-10:]))
                print_avg_comp = np.average(np.asarray(result_comp[-10:]))
            else:
                print_avg_trans = np.average(np.asarray(result_trans))
                print_avg_comp = np.average(np.asarray(result_comp))
            print("Final Avg Results for last 100 epoches: Avg. Trans Cost: {}, Avg. Compute Cost: {}".format(
                round(print_avg_trans, 2), round(print_avg_comp, 2)))
            print("----------------------------------------")

            writer.add_scalar('avg_cost/trans_cost',
                                round(print_avg_trans, 2), i_episode)
            writer.add_scalar('avg_cost/comp_cost',
                                round(print_avg_comp, 2), i_episode)
