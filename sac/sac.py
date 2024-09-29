import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import Critic, GaussianActor, DeterministicActor

class SAC(object):
    def __init__(self, local_dim, action_space, args):

        self.gamma = args.gamma # 折扣因子
        self.tau = args.tau # 目标网络软更新的混合系数
        self.alpha = args.alpha # 控制策略熵的权重，用于平衡探索和利用
        self.actor_type = args.policy # 策略类型
        self.target_update_interval = args.target_update_interval # 更新目标网络频率
        self.automatic_entropy_tuning = args.automatic_entropy_tuning # 是否自动调整熵的权重
        self.agent_num = args.server_num * args.ud_num
        self.LSTM = args.lstm
        self.global_info = args.global_info
        self.hidden_dim = args.hidden_size
        self.device = torch.device("cuda" if args.cuda else "cpu")

        # critic网络
        self.critic = Critic(local_dim, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = Critic(local_dim, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # actor网络
        if self.actor_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item() # 所有元素的乘积
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
                
            self.actor = GaussianActor(local_dim, args.hidden_size, action_space).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicActor(local_dim, args.hidden_size, action_space).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)
            
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ , _= self.actor.sample(1, state)
        return action.detach().cpu().numpy()[0]
        
    def select_action_info(self, i, states):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        local_state = states[i].unsqueeze(0)
        global_state = states[torch.arange(states.shape[0]) != i].unsqueeze(0)
        action, _, _ , _= self.actor.sample(2, local_state, global_state)
        return action.detach().cpu().numpy()[0]
        
    def select_action_lstm(self, index, states, state_seq, h_c):
        # [sequence_length, local_dim]
        state_seq = torch.FloatTensor(np.array(state_seq)).to(self.device)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        global_state = states[torch.arange(states.shape[0]) != index].unsqueeze(0)
        action, _, _, h_c = self.actor.sample(3, None, global_state, state_seq.unsqueeze(0), h_c)
        return action.detach().cpu().numpy()[0], h_c
    
    def update_parameters(self, index, memory, batch_size, updates, mold):
        # 状态、动作、奖励、下一状态、是否结束
        if mold == 3:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, state_seq_batch, next_state_seq_batch, old_hc_batch, hc_batch = memory.sample(mold, batch_size=batch_size)
            
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
            state_seq_batch = torch.FloatTensor(state_seq_batch).to(self.device)
            next_state_seq_batch = torch.FloatTensor(next_state_seq_batch).to(self.device)
            
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(mold, batch_size=batch_size)
            
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        # 计算next_q_value
        with torch.no_grad():
            if mold == 3:
                all_indices = torch.arange(next_state_batch.size(1))
                remaining_indices = all_indices[all_indices != index]
                next_global_state = next_state_batch[:, remaining_indices, :]
                
                next_action, next_log_prob, _, _ = self.actor.sample(mold, None, next_global_state, next_state_seq_batch, hc_batch)
                qf1_next_target, qf2_next_target = self.critic_target.forward_lstm(next_global_state, next_state_seq_batch, hc_batch, next_action)
            elif mold == 2:
                next_local_state = next_state_batch[:,index,:]
                all_indices = torch.arange(next_state_batch.size(1))
                remaining_indices = all_indices[all_indices != index]
                next_global_state = next_state_batch[:, remaining_indices, :]

                next_action, next_log_prob, _, _= self.actor.sample(mold, next_local_state, next_global_state)
                qf1_next_target, qf2_next_target = self.critic_target.forward_info(next_local_state, next_global_state, next_action)
            else:
                # 通过actor_target得到动作
                next_action, next_log_prob, _, _= self.actor.sample(mold, next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target.forward(next_state_batch, next_action)

            min_qf = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_prob
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf)

        # 更新critic
        if mold == 3:
            qf1, qf2= self.critic.forward_lstm(next_global_state, next_state_seq_batch, hc_batch, next_action)
        elif mold == 2:
            qf1, qf2= self.critic.forward_info(next_local_state, next_global_state, action_batch)
        else:
            qf1, qf2= self.critic.forward(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the actor improvement step           
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # 更新actor
        if mold == 3:
            pi, log_pi, _, _ = self.actor.sample(mold, None, next_global_state, next_state_seq_batch, hc_batch)
            qf1_pi, qf2_pi = self.critic.forward_lstm(next_global_state, next_state_seq_batch, hc_batch, pi)
        elif mold == 2:
            pi, log_pi, _, _ = self.actor.sample(mold, next_local_state, next_global_state)
            qf1_pi, qf2_pi = self.critic.forward_info(next_local_state, next_global_state, pi)
        else:
            pi, log_pi, _, _ = self.actor.sample(mold, state_batch)
            qf1_pi, qf2_pi = self.critic.forward(state_batch, pi)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)   
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
            
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            
        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])

            if evaluate:
                self.actor.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.actor.train()
                self.critic.train()
                self.critic_target.train()

