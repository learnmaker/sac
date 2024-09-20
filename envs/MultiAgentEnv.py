import inspect
import os
import sys
import math
import random
import numpy as np
from random import seed
from gym import spaces
import torch

# set parent directory as sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import chip_config, system_config, best_fDs
from tool.data_loader import load_data

tau = chip_config['tau'] # 最大容忍延迟 τ
fD = chip_config['fD'] # 单核计算频率
u = chip_config['u'] # 有效开关电容 μ
num_core = system_config['M'] 
num_task = system_config['F']
Cache = chip_config['C']
weight = system_config['weight'] 

MAX_STEPS = 1000


class MultiAgentEnv(object):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            init_sys_states,  # 初始系统状态, S^I, S^O for each task, plus A(0) in range [0, n_task-1]
            agent_num, # agent数量
            task_set,  # 任务集信息, for each sublist it is [I, O, w, τ]
            requests,  # the request task samples
            channel_snrs,   # a list of snr value of the channel
            exp_case
    ):
        super(MultiAgentEnv, self).__init__()
        self.task_set = task_set
        self.agent_num = agent_num
        self.current_step = 0
        self.global_step = 0
        self.channel_snrs = channel_snrs
        self.requests = requests
        self.sys_states = init_sys_states
        for agent_i in range(agent_num):
            self.sys_states[agent_i][-1] = self.requests[agent_i][self.global_step % len(self.requests[0])]
        self.init_sys_states = init_sys_states
        self._max_episode_steps = MAX_STEPS
        self.task_lists = [[] for _ in range(agent_num)] # 任务列表
        self.popularity = [[0] * num_task for _ in range(agent_num)]
        self.last_use = [[0] * num_task for _ in range(agent_num)]
        
        self.reactive_only = False
        self.no_cache = False
        self.heuristic = False
        self.best_fDs = None
        self.offload = False
        self.exp_case = exp_case
        
        self.not_vaild = 0
        print("所选的实验配置为: {}".format(exp_case))
        
        if exp_case == 'case1':  # case 1: no cache, reactive only, best fD choice (as baseline)
            self.reactive_only = True
            self.no_cache = True
            self.best_fDs = best_fDs
            print("无主动传输、无任务缓存、使用最佳计算核数")
        elif exp_case == 'case2':  # case 2: no cache, reactive only, dynamic fD
            self.reactive_only = True
            self.no_cache = True
            print("无主动传输、无任务缓存、使用动态计算核数")
        elif exp_case == 'case3':  # case 3: with cache, proactive transmission, dynamic fD
            print("主动传输、任务缓存、使用动态计算核数")
            pass
        elif exp_case == 'case4':  # case 4: with cache, reactive only, dynamic fD
            print("无主动传输、任务缓存、使用动态计算核数")
            self.reactive_only = True
        elif exp_case == 'case5':  # case 5: with cache, proactive transmission, dynamic fD, offload
            print("主动传输、任务缓存、使用动态计算核数、多agent计算卸载")
            self.offload = True
        elif exp_case == 'case6' or exp_case == 'case7':   # case 6, 7: MRU cache + LRU replace, MFU cache + LFU replace
            print("MRU cache + LRU replace, MFU cache + LFU replace")
            self.heuristic = True

        # 系统动作上下限 1 + F + F*2 + 1 计算核数、输入数据是否主动推送、缓存更新、卸载对象
        self.sample_low = np.asarray([-1] * (3 * num_task + 2), dtype=np.float32)
        self.sample_high = np.asarray([1] * (3 * num_task + 2), dtype=np.float32)
        
        # 系统状态上下限 F*2 + 1 缓存状态、请求
        self.observe_low = np.asarray([0]*num_task*2 + [0], dtype=np.float32)
        self.observe_high = np.asarray([1]*num_task*2 + [num_task-1], dtype=np.float32)
        
        self.action_space = spaces.Box(low=self.sample_low, high=self.sample_high, dtype=np.float32) # 系统动作空间
        self.observation_space = spaces.Box(low=self.observe_low, high=self.observe_high, dtype=np.float32) # 系统状态空间

        # [CR_At, b_f (all tasks), dSI_f (all tasks), dSO_f(all tasks)]
        if self.exp_case == 'case1':  # case 1: no cache, reactive only, best fD choice (as baseline)
            self.action_low = np.asarray([0] + [0]*num_task + [0]*num_task*2 + [-1])
            self.action_high = np.asarray([num_core] + [0]*num_task + [0]*num_task*2 + [-1])
            
        elif self.exp_case == 'case2':  # case 2: no cache, reactive only, dynamic fD
            self.action_low = np.asarray([0] + [0]*num_task + [0]*num_task*2 + [-1])
            self.action_high = np.asarray([num_core] + [0]*num_task + [0]*num_task*2 + [-1])
            
        elif self.exp_case == 'case3':  # case 3: with cache, proactive transmit, dynamic fD
            self.action_low = np.asarray([0] + [0]*num_task + [-1]*num_task*2 + [-1])
            self.action_high = np.asarray([num_core] + [1]*num_task + [1]*num_task*2 + [-1])
            
        elif self.exp_case == 'case4':  # case 4: with cache, reactive only, dynamic fD
            self.action_low = np.asarray([0] + [0]*num_task + [-1]*num_task*2 + [-1])
            self.action_high = np.asarray([num_core] + [0]*num_task + [1]*num_task*2 + [-1])
            
        elif self.exp_case == 'case5':  # case 5: with cache, proactive transmit, dynamic fD, offload
            self.action_low = np.asarray([0] + [0]*num_task + [-1]*num_task*2 + [-1])
            self.action_high = np.asarray([num_core] + [1]*num_task + [1]*num_task*2 + [agent_num-1])
            
        elif self.exp_case == 'case6' or exp_case == 'case7':  # case 6: with cache, most recently used cache, least recently used replace,
            # fixed computing cores
            self.action_low = np.asarray([int(num_core * 3 / 4)] + [0]*num_task + [0]*num_task*2 + [-1])
            self.action_high = np.asarray([int(num_core * 3 / 4)] + [0]*num_task + [0]*num_task*2 + [-1])
    
    def get_requests(self):
        return [row[-1] for row in self.sys_states]
    
    def get_cach_state(self):
        return [[row[i] for i in range(len(row) - 1)] for row in self.sys_states]
    
    def get_last_use(self):
        return self.last_use
    
    def show_detail2(self, detail2):
        print("被动传输消耗, 主动传输消耗, 被动计算消耗, 主动计算消耗")
        detail2 = np.column_stack(detail2)
        for i in range(self.agent_num):
            print("agent",i,detail2[i])
            
    def show_actions(self, actions):
        for index in range(self.agent_num):
            action = actions[index]
            action, prob_action = self.sample2action(action)
            print("agent",index,"action:",action)
        return
             
    def step(self, actions):
        self.current_step += 1
        self.global_step += 1
        dones = np.full(self.agent_num, False)
        new_actions=[]
        new_valid = True
        
        for index in range(self.agent_num):
            self.last_use[index][int(self.sys_states[index][-1])] = (self.current_step - 1) # 记录资源的最后一次使用时间
            self.popularity[index][int(self.sys_states[index][-1])] += 1 #记录资源的使用频率
            action = actions[index]
            action, prob_action = self.sample2action(action)
            valid, action = self.check_action_validity(index, action, prob_action)
            if valid == False:
                new_valid = False
            new_actions.append(action)
        
        # 计算传输消耗和计算消耗
        if self.offload:
            observation, observe_details, details2, prize = self.calc_observation_offload(new_actions)
        else:
            observation, observe_details, details2 = self.calc_observation(new_actions)
        for i in range(self.agent_num):
            if self.current_step > MAX_STEPS:
                dones[i] = True

        obs = self.next_state(new_actions, new_valid)
        # print("next_state",obs)
        self.sys_states = obs    # 更新系统状态
        total_cost_weight = 1
        # reward_ = - observation_ ** 2 / 1e12
        if self.offload:
            rewards = - (total_cost_weight * sum(observation + prize) + (1-total_cost_weight) * (observation)) / 1e6
        else:
            rewards = - (total_cost_weight * sum(observation) + (1-total_cost_weight) * observation) / 1e6
        actions = self.action2sample(new_actions)

        return self.scale_state(self.sys_states), rewards, dones, {'observe_details': observe_details, 'actions': actions}

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        self.current_step = 0
        # self.global_step -= 1
        self.sum_Comp = 0
        self.sum_Trans = 0
        self.popularity = [[0] * num_task for _ in range(self.agent_num)]
        self.last_use = [[0] * num_task for _ in range(self.agent_num)]
        self.sys_states = self.init_sys_states.copy()
        for agent_i in range(self.agent_num):
            self.sys_states[agent_i][-1] = self.requests[agent_i][self.global_step % len(self.requests[0])]
        self.task_lists = [[] for _ in range(self.agent_num)]
        return self.scale_state(self.sys_states)

    def render(self, mode='human', close=False):
        """Render the environment to the screen"""
        print(f'Step: {self.global_step}')

    # 返回系统状态
    def system_state(self):
        return self.sys_states.copy()
    
    def calc_observation(self, actions):
        total_B_R=np.full(self.agent_num, 0.)
        total_B_P=np.full(self.agent_num, 0.)
        total_E_R=np.full(self.agent_num, 0.)
        total_E_P=np.full(self.agent_num, 0.)
        
        for agent_i in range(self.agent_num):
            A_t = int(self.sys_states[agent_i][-1])
            snr_t = self.channel_snrs[agent_i][self.global_step % len(self.channel_snrs)]
            I_At = self.task_set[A_t][0]
            w_At = self.task_set[A_t][2]
            S_I_At = self.sys_states[agent_i][A_t]
            S_O_At = self.sys_states[agent_i][num_task + A_t]
            C_R_At = actions[agent_i][0]

            if S_I_At == 1 or S_O_At == 1:
                B_R = 0
            else:
                B_R = (1 - S_I_At) * (1 - S_O_At) * I_At / ((tau - I_At * w_At / (C_R_At * fD)) * math.log2(1 + snr_t))
            E_R = (1 - S_O_At) * u * (C_R_At * fD) ** 2 * I_At * w_At
            # print("延迟时间",tau)
            # print("剩余时间",tau - I_At * w_At / (C_R_At * fD))
            
            total_B_R[agent_i] = B_R
            total_E_R[agent_i] = E_R
            if self.reactive_only:
                continue

            E_P = 0
            B_P = 0
            for idx in range(num_task):
                C_P = 0
                E_P += u * (C_P * fD) ** 2 * self.task_set[idx][0] * self.task_set[idx][2]  # always 0 since C_P is zero
                B_P += self.task_set[idx][0] * actions[agent_i][1 + idx] / (tau * math.log2(1 + snr_t))
            
            total_E_P[agent_i] = E_P
            total_B_P[agent_i] = B_P
        
        return total_B_R + total_B_P + (total_E_R + total_E_P) * weight, [total_B_R + total_B_P, total_E_R + total_E_P], [total_B_R, total_B_P, total_E_R, total_E_P]
    
    # 计算传输消耗和计算消耗
    def calc_observation_offload(self, actions):
        """
        # action: [CR_At, CP_f (all tasks), b_f (all tasks), dSI_f (all tasks), dSO_f(all tasks), O_u, O_m]
        # object: calculate B_R(被动传输带宽) + B_P(主动传输带宽) + E_R(计算消耗) + E_P()
        """
        total_B_R=np.full(self.agent_num, 0.)
        total_B_P=np.full(self.agent_num, 0.)
        total_E_R=np.full(self.agent_num, 0.)
        total_E_P=np.full(self.agent_num, 0.)

        total_prize = np.full(self.agent_num, 0.)
        total_I = np.full(self.agent_num, 0.)
        C_time = np.full(self.agent_num, 0.)

        # 更新任务列表
        for agent_i in range(self.agent_num):
            # 当前agent产生的任务
            A_t = int(self.sys_states[agent_i][-1])
            # 卸载对象
            O = int(actions[agent_i][-1])
            if O == -1:
                self.task_lists[agent_i]=[[-1, A_t]] + self.task_lists[agent_i]
            else:
                self.task_lists[O].append([agent_i, A_t]) # agent_i 卸载给O的A_t任务
                
        # print("任务列表", self.task_lists)
        # self.showActions(actions)
        
        # 对每个agent
        for agent_i in range(self.agent_num):
            snr_local = self.channel_snrs[agent_i][self.global_step % len(self.channel_snrs)]
            # 主动传输
            for idx in range(num_task):
                C_P = 0
                total_E_P[agent_i] += u * (C_P * fD) ** 2 * self.task_set[idx][0] * self.task_set[idx][2]  # always 0 since C_P is zero
                total_B_P[agent_i] += self.task_set[idx][0] * actions[agent_i][1 + idx] / (tau * math.log2(1 + snr_local))

            # 任务列表
            agent_i_tasks = self.task_lists[agent_i]
            
            # 计算核数
            C_R_At = actions[agent_i][0]

            # 如果任务列表不为空
            if agent_i_tasks:
                flag = False # 是否溢出
                local_At = -1 # 本地计算任务
                local = False # 是否为本地计算任务
                # 对agent的每个任务
                for task in agent_i_tasks:
                    offload_agent, offload_A_t = task
                    
                    if offload_agent == -1:
                        local = True
                        offload_agent = agent_i
                    else:
                        local = False
                    if flag:
                        total_prize[offload_agent] += 99999
                        # print("无法得到卸载反馈结果，+99999")
                        continue
                    
                    E_R=0
                    B_R=0                    
                    
                    I_At = self.task_set[offload_A_t][0] # 输入数据大小
                    O_At = self.task_set[offload_A_t][1] # 输出数据大小
                    w_At = self.task_set[offload_A_t][2] # 每比特所需计算周期
                    # 对应任务缓存状态
                    S_I_At = self.sys_states[agent_i][offload_A_t]
                    S_O_At = self.sys_states[agent_i][num_task + offload_A_t]
                    
                    # 如果有输出缓存，无需计算、无需传输
                    if S_O_At == 1 or offload_A_t == local_At:
                        E_R = 0
                        B_R = 0
                        
                    # 否则，需要计算
                    else:
                        C_time[agent_i] += I_At * w_At / (C_R_At * fD)
                        # 如果计算时间大于最大容许延迟
                        if C_time[agent_i] >= tau:
                            # 惩罚卸载用户
                            total_prize[offload_agent] += 99999
                            C_time[agent_i] -= I_At * w_At / (C_R_At * fD)
                            flag = True
                            continue
                        
                        E_R = (1 - S_O_At) * u * (C_R_At * fD) ** 2 * I_At * w_At
                            
                        # 如果没有输入数据，需要从服务器下载数据
                        if S_I_At == 0 and offload_A_t != local_At:
                            total_I[agent_i] += I_At
                            # print("从服务器下载数据, +",I_At)
                            B_R += I_At
                            
                    # 最后，如果是卸载任务，需要回传
                    if not local:
                        total_I[agent_i] += O_At
                        B_R += O_At
                        total_I[offload_agent] += O_At
                        # print("计算结果回传, +",O_At)
                    else:
                        local_At = offload_A_t
                        
                    total_E_R[agent_i] += E_R
        
        for agent_i in range(self.agent_num):
            snr_local = self.channel_snrs[agent_i][self.global_step % len(self.channel_snrs)]
            # 处理完所有任务，计算被动传输带宽
            total_B_R[agent_i] = total_I[agent_i] / ((tau - C_time[agent_i]) * math.log2(1 + snr_local))
            # total_prize[agent_i] = (prize_E[agent_i] * weight + prize_B[agent_i] / ((tau - C_time[agent_i]) * math.log2(1 + snr_local))) * 1.2

        total_B_R = np.array(total_B_R)
        total_E_R = np.array(total_E_R)
        total_B_P = np.array(total_B_P)
        total_E_P = np.array(total_E_P)

        # 仅被动传输
        if self.reactive_only:
            return total_B_R + total_E_R * weight, [total_B_R, total_E_R], [total_B_R, 0, total_E_R, 0], total_prize

        return total_B_R + total_B_P + (total_E_R + total_E_P) * weight, [total_B_R + total_B_P, total_E_R + total_E_P], [total_B_R, total_B_P, total_E_R, total_E_P], total_prize

    # 样本到动作空间
    def sample2action(self, action):
        unit = [2.0 / (self.action_high[idx] - self.action_low[idx] + 1) for idx in range(len(self.action_high))]
        rescale_action = []
        prob_action = []
        for ide, elem in enumerate(action):
            rescale_action.append(min(self.action_low[ide] + (elem + 1) // unit[ide], self.action_high[ide]))
            prob_action.append((elem + 1) / 2)

        return rescale_action, prob_action
    
    # 动作空间到样本
    def action2sample(self, action):
        unit = [2.0 / (self.action_high[idx] - self.action_low[idx] + 1) for idx in range(len(self.action_high))]
        sample = []
        for ide, elem in enumerate(action):
            sample.append((elem - self.action_low[ide] + 0.5) * unit[ide] - 1)

        return sample

    # 归一化
    def scale_state(self, state):
        # Scale to [-1, 1]
        length = self.observe_high - self.observe_low
        scaled_state = []
        for idx, elem in enumerate(state):
            scaled_state.append(2 * (elem - self.observe_low[idx]) / length - 1)
        return scaled_state

    def next_state(self, actions, valid=True):
        """
        action: [CR_At, b_f (all tasks), dSI_f (all tasks), dSO_f(all tasks)]
        calculate the system state for t+1, S_I(f), S_O(f); note that the A(t+1) is set as 0 which needs to be updated
        S_I(f, t+1) = S_I(f, t) + dS_I(f, t)
        S_O(f, t+1) = S_O(f, t) + dS_O(f, t)
        """
        if not valid:
            # Not do update when the action is not valid
            self.not_vaild += 1
            print("动作不合格，不更新系统状态")
            return self.sys_states

        next_states =  [[0] * len(self.sys_states[0]) for _ in range(self.agent_num)]
        # 更新S(t)
        for agent_i in range(self.agent_num):
            for idx in range(num_task):
                S_I = self.sys_states[agent_i][idx]
                S_O = self.sys_states[agent_i][num_task + idx]
                dS_I = actions[agent_i][1 + num_task + idx]
                dS_O = actions[agent_i][1 + num_task * 2 + idx]
                next_states[agent_i][idx] = S_I + dS_I
                next_states[agent_i][num_task + idx] = S_O + dS_O
                assert 0 <= (S_I + dS_I) <= 1 and 0 <= (S_O + dS_O) <= 1

            next_states[agent_i][-1] = self.requests[agent_i][self.global_step % len(self.requests)]
        return np.array(next_states)

    # 单个agent
    def check_action_validity(self, agent_i, action, prob_action):
        # print(action, prob_action)
        """
        Input:
            action: [CR_At, b_f (all tasks), dSI_f (all tasks), dSO_f(all tasks)] 计算核数、主动传输、缓存更新
            sys_states: [S_I(f) (all tasks), S_O(f) (all tasks), At], where At = [0, F-1] 缓存状态、请求任务
        Constraints:
            1) I(f) * w(f) / tau <= M * fD     # system constraint (not check here)
            2) I(At) * w(At) / tau <= C_R(At) * fD,     when S_O(At)=0
            3) C_R(f) = 0,  for all f not At, or S_O(f)=1
            4) I(f) * w(f) / tau <= C_P(f) * fD,    when C_P(f) > 0
            5) C_P(f) <= S_I(f) * M
            6) sum of C_R(f) + C_P(f) <= M
            7) -S_I(f) <= dS_I(f) <= min{b(f), 1-S_I(f)}
            8) -S_O(f) <= dS_O(f) <= min{C_R(f)+C_P(f), 1-S_O(f)}
            9) sum of I(f) * (S_I(f) + dS_I(f)) + O(f) * (S_O(f) +dS_O(f)) <= C
        """
        # 数据准备
        A_t = int(self.sys_states[agent_i][-1]) # 请求任务
        S_O_At = self.sys_states[agent_i][num_task + A_t] # 请求任务输出缓存
        I_At = self.task_set[A_t][0] # 请求任务输入大小
        w_At = self.task_set[A_t][2] # 每比特所需的计算周期
        CR_At = action[0] # 分配计算核数

        b_f = action[1:1 + num_task].copy() # 主动传输决策
        dS_I_f = action[1 + num_task:1 + num_task * 2].copy()
        dS_O_f = action[1 + num_task * 2:1 + num_task * 3].copy()

        b_f_prob = prob_action[1:1 + num_task].copy()
        dS_I_f_prob = prob_action[1 + num_task:1 + num_task * 2].copy()
        dS_O_f_prob = prob_action[1 + num_task * 2:1 + num_task * 3].copy()

        S_I_f = self.sys_states[agent_i][:num_task].copy() # 输入缓存
        S_O_f = self.sys_states[agent_i][num_task:num_task * 2].copy() # 输出缓存
        I_f = [self.task_set[idx][0] for idx in range(num_task)] # 任务输入大小
        O_f = [self.task_set[idx][1] for idx in range(num_task)] # 任务输出大小
        # w_f = [self.task_set[idx][2] for idx in range(num_task)]

        b_f_new = [0] * num_task
        dS_I_f_new = [0] * num_task
        dS_O_f_new = [0] * num_task
        # ---------------------数据准备结束-------------------
        # self.showAction(action)
        
        # 1. 纠正计算核数 I(At) * w(At) / tau <= C_R(At) * fD
        if I_At * w_At / tau > (CR_At * fD) and S_O_At == 0: # 没有输出缓存，同时分配计算核数不够
            CR_At = min(math.ceil(I_At * w_At / tau / fD), num_core)
            # print("分配计算核数不够",CR_At, num_core)
        elif S_O_At == 1:
            CR_At = 0

        # choose best fD for reactive processing (only for case 1)
        if self.best_fDs is not None:
            CR_At = self.best_fDs[A_t]
        C_R_f = [0] * num_task
        C_R_f[A_t] = CR_At
        action[0] = int(CR_At)

        # 对于没有缓存的方案，只纠正了计算核数
        if self.no_cache:
            return True, action

        # 对于启发式方案
        if self.heuristic:
            if self.exp_case == 'case6':    # MRU cache + LRU replace
                # cache the input data of most recently used if it has not been cached
                most_interest_A = np.argmin(np.abs(np.asarray(self.last_use) - self.current_step))
                indic = np.where(S_I_f == 1)[0]
                least_interest_A = None
                if indic.size > 0:
                    least_interest_A = indic[np.argmax(np.abs(np.asarray(self.last_use)[indic] - self.current_step))]

            elif self.exp_case == 'case7':  # MFU cache + LFU replace
                # cache the input data of most frequently used if it has not been cached
                most_interest_A = np.argmax(np.asarray(self.popularity))
                indic = np.where(S_I_f == 1)[0]
                least_interest_A = None
                if indic.size > 0:
                    least_interest_A = indic[np.argmin(np.asarray(self.popularity)[indic])]

            if S_I_f[most_interest_A] == 0:
                dS_I_f_new[most_interest_A] = 1
                if most_interest_A != A_t:
                    b_f_new[most_interest_A] = 1  # 主动传输
                is_cache_exceed = self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, dS_I_f_new, dS_O_f_new)
                # remove the cache for least frequently used task if it has cache
                if is_cache_exceed and least_interest_A is not None:
                    if S_I_f[least_interest_A] == 1:
                        dS_I_f_new[least_interest_A] = -1

            is_cache_exceed = self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, dS_I_f_new, dS_O_f_new)
            action[1:1 + num_task] = b_f_new.copy()
            action[1 + num_task:1 + num_task * 2] = dS_I_f_new.copy()

            return not is_cache_exceed, action

        # below are for non-heuristic, SAC solution
        bf_sort_idx = np.argsort(b_f_prob)[::-1] # 从概率大的向概率小的排序

        # 概率最大的主动传输任务 + 输入输出缓存都为0
        if b_f[bf_sort_idx[0]] > 0 and S_I_f[bf_sort_idx[0]] + S_O_f[bf_sort_idx[0]] < 1:
            b_f_new[bf_sort_idx[0]] = 1

        # 主动传输从概率变为具体的动作
        if not self.reactive_only:
            action[1:1 + num_task] = b_f_new.copy()

        push_idx = [idp for idp in range(num_task)] + [idp for idp in range(num_task)]
        push_IO_indc = [0] * num_task + [1] * num_task   # 0 for S_I and 1 for S_O
        push_prob = dS_I_f_prob + dS_O_f_prob

        for idx, b in enumerate(b_f_new):
            if b == 1:
                # when b = 1, dS_I only be 1, dS_O >= 0, best policy 3
                dS_I_f_new[idx] = 1
                # dS_O_f_new[idx] = 0

        # Constraint (9) sum of I(f) * (S_I(f) + dS_I(f)) + O(f) * (S_O(f) +dS_O(f)) <= C
        is_cache_exceed = self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, dS_I_f_new, dS_O_f_new)

        # 超过
        if is_cache_exceed:
            # 优先清理概率最小的
            drop_sort = np.argsort(push_prob)
            for idx in list(drop_sort):
                if b_f_new[push_idx[idx]] > 0:
                    continue
                # 存在输入缓存
                if push_IO_indc[idx] == 0 and S_I_f[push_idx[idx]] > 0:
                    dS_I_f_new[push_idx[idx]] = -1
                # 存在输出缓存
                elif push_IO_indc[idx] == 1 and S_O_f[push_idx[idx]] > 0:
                    dS_O_f_new[push_idx[idx]] = -1
                is_cache_exceed = self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, dS_I_f_new, dS_O_f_new)
                if not is_cache_exceed: # 如果超出缓存，继续清理
                    break
        # 没超过
        if not is_cache_exceed:
            push_sort = np.argsort(push_prob)[::-1]
            for idx in list(push_sort):
                if b_f_new[push_idx[idx]] > 0:
                    continue
                if push_IO_indc[idx] == 0 and S_I_f[push_idx[idx]] < 1 and C_R_f[push_idx[idx]] > 0:
                    dS_I_f_new[push_idx[idx]] = 1
                elif push_IO_indc[idx] == 1 and S_O_f[push_idx[idx]] < 1 and C_R_f[push_idx[idx]] > 0:
                    dS_O_f_new[push_idx[idx]] = 1
                is_cache_exceed = self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, dS_I_f_new, dS_O_f_new)
                if is_cache_exceed:
                    # convert it back for this change
                    if push_IO_indc[idx] == 0 and S_I_f[push_idx[idx]] < 1 and C_R_f[push_idx[idx]] > 0:
                        dS_I_f_new[push_idx[idx]] = 0
                    elif push_IO_indc[idx] == 1 and S_O_f[push_idx[idx]] < 1 and C_R_f[push_idx[idx]] > 0:
                        dS_O_f_new[push_idx[idx]] = 0
                    break

        # give the action correction for non-reactive-only methods
        if not self.reactive_only:
            action[1 + num_task:1 + num_task * 2] = dS_I_f_new.copy()
            action[1 + num_task * 2:1 + num_task * 3] = dS_O_f_new.copy()
        else:
            for idx, b in enumerate(b_f):
                dS_I_f[idx] = max(-S_I_f[idx], min(dS_I_f[idx], min(C_R_f[idx] + b, 1 - S_I_f[idx])))
                dS_O_f[idx] = max(-S_O_f[idx], min(dS_O_f[idx], min(C_R_f[idx], 1 - S_O_f[idx])))
            action[1 + num_task:1 + num_task * 2] = dS_I_f.copy()
            action[1 + num_task * 2:1 + num_task * 3] = dS_O_f.copy()

        # print(action[1 + num_task:1 + num_task * 2], action[1 + num_task * 2:1 + num_task * 3], self.sys_states)

        if self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, action[1+num_task:1+num_task*2], action[1+num_task*2:1+num_task*3]):
            print("超过缓存容量")
            return False, action
        else:
            return True, action

    # 检测是否超过缓存容量
    def test_cache_exceed(self, I_f, O_f, S_I_f, S_O_f, dS_I_f, dS_O_f):
        sum_cache = np.sum(np.asarray(I_f) * (np.asarray(S_I_f) + np.asarray(dS_I_f)) +
                           np.asarray(O_f) * (np.asarray(S_O_f) + np.asarray(dS_O_f)))
        # print("需要缓存容量", sum_cache)
        # print("缓存容量",Cache)
        if sum_cache > Cache:
            is_cache_exceed = True
        else:
            is_cache_exceed = False

        return is_cache_exceed

    def getCurrent_step(self):
        return self.current_step
    
    def showActions(self, actions):
        for i, action in enumerate(actions):
            print("agent",i,"动作：",action[0],action[1:num_task+1],action[num_task+1:2*num_task+1],action[2*num_task+1:3*num_task+1],action[-1])
            
    def showAction(self, action):
        print("动作：",action[0],action[1:num_task+1],action[num_task+1:2*num_task+1],action[2*num_task+1:3*num_task+1],action[-1])
        
    def getNotValid(self):
        return self.not_vaild


