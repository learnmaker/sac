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
from multiprocessing import Process, Pipe, Manager


# 命令行参数设置
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
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
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
# 熵前面的温度系数
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
# 熵前面的系数是否自动调整
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
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
# 多少次随机采样
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling1 random actions (default: 10000)')
# 目标网络的更新周期
parser.add_argument('--target_update_interval', type=int, default=1000, metavar='N',
                    help='Value target update per no. of updates per step (default: 1000)')
# 经验缓冲区大小
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
# 是否使用cuda
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
# 服务器个数
parser.add_argument('--server_num', type=int, default=2,
                    help='server number(default: 2)')
# 每个服务器的用户设备个数
parser.add_argument('--ud_num', type=int, default=3,
                    help='user device number(default: 3)')

args = parser.parse_args()


# 环境设置
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# 中央管理器(保存所有用户设备的任务请求，2 * 3 二维矩阵)
CM = np.full((args.server_num, args.ud_num), -1)

# 服务器 (每台服务器存储全局缓存信息，2 * 3 * 6 * 2)
task_num = system_config['F']  # 任务数 6
maxp = system_config['maxp']   # 最大转移概率 70%
task_utils = load_data('./mydata/task_info/task'+str(task_num) + '_utils.csv')  # 任务集信息[I, O, w，τ]
task_set_ = task_utils.tolist()
servers = [np.full((args.server_num, args.ud_num, task_num, 2), 0) for _ in range(args.server_num)]

# 跟据服务器数量和用户设备数量生成 任务请求和信噪比，保存在temp文件夹
generate_snrs(args.server_num) # 生成信噪比
generate_request(args.server_num, args.ud_num, task_num, maxp) # 生成任务请求



sys.exit()
channel_snrs = []
for i in range(args.server_num):
    snr=load_data('./mydata/temp/dynamic_snrs_'+str(i+1)+'.csv').T
    channel_snrs.append(snr)

Ats = []
for server in range(args.server_num):
    s_uds=[]
    for ud in range(args.ud_num):
        At=load_data("./mydata/temp/server"+str(server+1)+"_ud"+str(ud+1)+"_samples"+str(task_num)+"_maxp"+str(maxp)+".csv").T
        s_uds.append(At)
    Ats.append(s_uds)

envs=[]
for server in range(args.server_num):
    s_envs=[]
    for ud in range(args.ud_num):
        # 任务缓存状态[S^I, S^O, A(0)]、任务信息、任务请求、信噪比 (server_num * ud_num)
        env = MultiTaskCore(init_sys_state=[0] * (2 * task_num) + [1], task_set=task_set_, requests=Ats[server][ud],
                    channel_snrs=channel_snrs[server], exp_case=args.exp_case)
        # env.seed(args.seed)
        env.action_space.seed(args.seed)
        s_envs.append(env)
    envs.append(s_envs)

# 用户设备Agents (server_num * ud_num)
agents = []
for server in range(args.server_num):
    s_agent=[]
    for ud in range(args.ud_num):
        agent = SAC(envs[server][ud].observation_space.shape[0], envs[server][ud].action_space, args)
        s_agent.append(agent)
    agents.append(s_agent)
    
# 经验缓存区 (server_num * ud_num)
memories=[]
for server in range(args.server_num):
    s_memory=[]
    for ud in range(args.ud_num):
        memory = ReplayMemory(args.replay_size, args.seed)
        s_memory.append(memory)
    memories.append(s_memory)

# Tensorboard
writer = SummaryWriter(
    'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                  args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# 训练
total_numsteps = 0
updates = 0
result_trans = []
result_comp = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    # 训练到结束状态
    while not done:
        # 对每个agent进行一次训练
        for agent, memory in zip(agents, memories):
            # 前start_steps随机探索
            if args.start_steps > total_numsteps:
                # Sample random action
                action = env.action_space.sample()
            else:
                # Sample action from policy
                action = agent.select_action(state)

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                        memory, args.batch_size, updates)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar(
                        'entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, info = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(
            not done)
        memory.push(state, action, reward, next_state, mask)
        state = next_state

    # 大于最大训练次数，退出
    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                  episode_steps,
                                                                                  round(episode_reward, 2)))
    
    # 每10个轮次评估模型的好坏
    eval_freq = 10
    # eval_freq = 1
    if i_episode % eval_freq == 0 and args.eval is True:
        avg_reward = 0.
        avg_trans_cost = 0.
        avg_compute_cost = 0.
        episodes = 10
        done_step = 0
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            trans_cost = 0
            compute_cost = 0
            done = False
            while not done:
                # print(env.sys_state)
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                try:
                    trans_cost += info['observe_detail'][0]
                    compute_cost += info['observe_detail'][1]
                    done_step += 1
                except:
                    pass
                state = next_state

            avg_reward += episode_reward
            avg_trans_cost += trans_cost
            avg_compute_cost += compute_cost

        avg_reward /= episodes
        avg_trans_cost /= done_step
        avg_compute_cost /= done_step
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

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
        writer.add_scalar('avg_cost/trans_cost',
                          round(print_avg_trans, 2), i_episode)
        writer.add_scalar('avg_cost/comp_cost',
                          round(print_avg_comp, 2), i_episode)
        # raise

# env.close()
