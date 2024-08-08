import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma # 折扣因子
        self.tau = args.tau # 目标网络软更新的混合系数
        self.alpha = args.alpha # 控制策略熵的权重，用于平衡探索和利用

        self.policy_type = args.policy # 策略类型
        self.target_update_interval = args.target_update_interval # 更新目标网络频率
        self.automatic_entropy_tuning = args.automatic_entropy_tuning # 是否自动调整熵的权重

        self.device = torch.device("cuda" if args.cuda else "cpu")

        # 价值网络 state + action--> score
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device) # 主Q网络，用于评估状态-动作对的价值
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr) # Q网络的优化器，使用Adam优化算法
        
        # 目标价值网络(更新较慢)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device) # 目标Q网络，用于稳定学习过程，减少训练波动
        hard_update(self.critic_target, self.critic) # 初始时，目标网络的权重被硬拷贝（完全复制）自主网络的权重

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item() # 所有元素的乘积
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            # 策略网络（actor）state-->action
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            # 策略网络
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, server_requests, servers_cache_states, evaluate=False):
        tensor_state = torch.FloatTensor(state)
        tensor_request = torch.FloatTensor(server_requests)
        tensor_cach = torch.FloatTensor(servers_cache_states)
        state = torch.cat((tensor_state, tensor_request, tensor_cach.view(-1)), dim=0).to(self.device).unsqueeze(0) # 拼接矩阵
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # 状态、动作、奖励、下一状态、是否结束
        state_batch, old_request_batch, old_cach_batch, action_batch, reward_batch, next_state_batch, request_batch, cach_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        old_request_batch = torch.FloatTensor(old_request_batch).to(self.device)
        old_cach_batch = torch.FloatTensor(old_cach_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        request_batch = torch.FloatTensor(request_batch).to(self.device)
        cach_batch = torch.FloatTensor(cach_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        state = torch.cat((state_batch, old_request_batch, old_cach_batch.view(batch_size,-1)), dim=1)
        next_state = torch.cat((next_state_batch, request_batch, cach_batch.view(batch_size,-1)), dim=1)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic(state, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state)

        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

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

        # critic损失、actor损失、温度参数α的损失和温度参数α的值
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

