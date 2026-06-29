# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

# --- 辅助函数：计算折扣累积回报 G_t ---
def compute_discounted_return(token_level_rewards: torch.Tensor, gamma: float):
    """
    Computes Monte Carlo discounted return G_t from the rewards.
    G_t = R_t + gamma * R_{t+1} + gamma^2 * R_{t+2} + ...
    
    Args:
        token_level_rewards: shape (response_length)
        gamma: Discount factor
    Returns:
        discounted_returns: shape (response_length)
    """
    gen_len = token_level_rewards.shape[-1]
    returns = torch.zeros_like(token_level_rewards)
    
    # 使用累积求和（cumsum）的变体或循环进行向后计算，以确保时序性。
    # 这里使用循环来实现 G_t 的定义：
    current_return = 0.0
    for t in reversed(range(gen_len)):
        current_return = token_level_rewards[t] + gamma * current_return
        returns[t] = current_return
    return returns

# --- 向量化辅助函数：计算折扣累积回报 G_t ---
def compute_discounted_return_vectorized(rewards_masked: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    完全向量化计算 G_t = R_t + gamma * R_{t+1} + ... (Batch-wise Vectorization)
    """
    bsz, T = rewards_masked.shape
    device = rewards_masked.device
    dtype = rewards_masked.dtype

    # 1. 构造 gamma 幂次序列: [1, gamma, gamma^2, ..., gamma^{T-1}]
    powers = torch.arange(T, device=device, dtype=dtype)
    gamma_powers = torch.pow(gamma, powers) 

    # 2. 构造衰减因子序列: [gamma^{T-1}, gamma^{T-2}, ..., 1]
    decay_factors = torch.flip(gamma_powers, dims=[0]).unsqueeze(0) # shape (1, T)
    
    # 3. 缩放奖励: R' = R_t * gamma^{T-1-t}
    R_scaled = rewards_masked * decay_factors

    # 4. 前向累积求和: CumSum(R')
    cumsum_R_scaled = torch.cumsum(R_scaled, dim=-1)

    # 5. 反向还原折扣: G'_t / gamma^{T-1-t}
    returns_scaled = cumsum_R_scaled / (decay_factors + 1e-8) 

    # 6. 反转时序: 得到正确的 G_t 形式
    returns = torch.flip(returns_scaled, dims=[1])

    return returns

# --- 向量化辅助函数：计算折扣累积回报 G_t ---
def compute_discounted_return_iterative(rewards_masked: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    使用反向迭代法计算 G_t = R_t + gamma * G_{t+1}
    适用于长序列，O(T) 且内存友好。
    """
    bsz, T = rewards_masked.shape
    returns = torch.zeros_like(rewards_masked)
    # 初始化 G_{T-1}
    returns[:, T - 1] = rewards_masked[:, T - 1]
    
    # 反向迭代
    for t in range(T - 2, -1, -1):
        # 向量化计算：针对整个批次 BZ
        returns[:, t] = rewards_masked[:, t] + gamma * returns[:, t + 1]
        
    return returns

# --- 向量化辅助函数：计算折扣累积回报 G_t ---
def compute_discounted_return_iterative_sparse(rewards_masked: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    使用反向迭代法计算 G_t = R_t + gamma * G_{t+1}
    适用于长序列，O(T) 且内存友好。
    """
    bsz, T = rewards_masked.shape
    returns = torch.zeros_like(rewards_masked)
    # 初始化 G_{T-1}
    returns[:, T - 1] = rewards_masked[:, T - 1]
    
    # 反向迭代
    for t in range(T - 2, -1, -1):
        # 向量化计算：针对整个批次 BZ
        returns[:, t] = rewards_masked[:, t] + gamma * returns[:, t + 1]
    # 判定哪些位置的原始奖励 R_t 不为零
    non_zero_mask = (rewards_masked != 0)
    
    # 使用掩码过滤 returns：只有 non_zero_mask 为 True 的位置保留 returns 的值，
    # 否则为 0。
    returns_sparse = returns * non_zero_mask.float()
        
    return returns, returns_sparse

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_progress_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   gamma: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    import time
    time0 = time.time()
    bsz, response_length = token_level_rewards.shape
        
    # ------------------------------------------------------------------
    # Step 1: 计算 G_token (时序累积回报) 和 G_episode (总回报)
    # ------------------------------------------------------------------
    # 对每个样本计算 G_token (时序累积回报)
    # token_level_return = torch.zeros_like(token_level_rewards)
    # with torch.no_grad():
    #     for i in range(bsz):
    #         sample_rewards = token_level_rewards[i]
    #         # 确保只计算到 EOS 标记
    #         if eos_mask is not None:
    #             valid_len = (eos_mask[i] == 1).sum()
    #             sample_rewards = sample_rewards[:valid_len]
    #         if sample_rewards.numel() > 0:
    #             token_level_return[i, :sample_rewards.shape[-1]] = compute_discounted_return(sample_rewards, gamma)
    # # G_episode (总回报, G_0): 取每个样本的第一个 Token 的 G_token 值 (G_0)
    # g_episode = token_level_return[:, 0].clone()
    
    token_level_return = torch.zeros_like(token_level_rewards)
    with torch.no_grad():
        
        # 1. 奖励掩码 (处理稀疏性和 EOS)
        # 只需要将填充部分的奖励设为 0，然后向量化函数会处理整个批次
        rewards_masked = token_level_rewards * eos_mask
        
        # 2. 调用向量化函数
        token_level_return = compute_discounted_return_vectorized(rewards_masked, gamma)
        
        # G_episode (总回报, G_0)
        g_episode = token_level_return[:, 0].clone()
            
    time1 = time.time()
    # ------------------------------------------------------------------
    # Step 2: 计算 A_GRPO_group (组优势)
    # ------------------------------------------------------------------
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(g_episode[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # 计算优势函数
        a_grpo_scalar = torch.zeros_like(g_episode)
        for i in range(bsz):
            idx = index[i]
            # (G_episode - mu) / sigma
            a_grpo_scalar[i] = (g_episode[i] - id2mean[idx]) / (id2std[idx] + epsilon)
    time2 = time.time()
    # ------------------------------------------------------------------
    # Step 3: 计算最终的加权优势 A_token
    # ------------------------------------------------------------------
    # # A_token = (G_token / G_episode) * A_GRPO_group
    # g_episode_tiled = g_episode.unsqueeze(-1).expand_as(token_level_return)
    # g_episode_abs = torch.abs(g_episode_tiled)
    # weight_factor = (torch.abs(token_level_return) / (g_episode_abs + epsilon)) * eos_mask
    # weight_factor = torch.clamp(weight_factor, max=1.0)
    # # weight_factor = (token_level_return / (g_episode_tiled + epsilon)) * eos_mask
    # # weight_factor = torch.clamp(weight_factor, max=1.0)
    # a_grpo_tiled = a_grpo_scalar.unsqueeze(-1).expand_as(token_level_return)
    # advantages = weight_factor * a_grpo_tiled * eos_mask
    # returns = g_episode_tiled * eos_mask
    # time3 = time.time()
    # # print("time1", time1-time0)
    # # print("time2", time2-time1)
    # # print("time3", time3-time2)
    
    with torch.no_grad():
        # 1. 找到 Gt 集合的实际最小值 Gt_t_min = min(Gt)
        # Gt_t_min_per_seq: shape (bs, 1)
        Gt_t_min_per_seq, _ = token_level_return.min(dim=-1, keepdim=True) 
        
        # 2. 修正基线 Gmin: Gmin = min(Gt_t_min, 0)
        # Gmin_scalar: shape (bs, 1)
        Gmin_scalar = torch.min(Gt_t_min_per_seq, torch.tensor(0.0, device=token_level_return.device))
        
        # 将标量 (G0) 和 Gmin 扩展为 (bs, response_length)
        G_tiled = g_episode.unsqueeze(-1).expand_as(token_level_return)
        Gmin = Gmin_scalar.expand_as(token_level_return)
        
        # 3. 计算 Weight: weight = clamp((Gt - Gmin) / (G - Gmin), 0, 1)
        # weight: (Gt - Gmin) / (G0 - Gmin); Clamp: clamp(..., 0, 1)
        denominator = G_tiled - Gmin + epsilon
        weight_factor = (token_level_return - Gmin) / denominator
        weight = torch.clamp(weight_factor, min=0.0, max=1.0)
        
        # 4. 准备 GRPO 优势的平铺版本和掩码
        # 创建用于分段计算的掩码
        a_grpo_scalar_expanded = a_grpo_scalar.unsqueeze(-1).expand_as(token_level_return)
        positive_mask = (a_grpo_scalar_expanded >= 0).float()
        negative_mask = 1.0 - positive_mask
    
    # 5. 分段计算 advantages (在 no_grad() 外计算，如果需要梯度流向 pi)
    # A_t = weight * A_GRPO_group  (if A_GRPO >= 0)
    # A_t = (1 - weight) * A_GRPO_group  (if A_GRPO < 0)
    advantages_positive = weight * a_grpo_scalar_expanded * positive_mask
    advantages_negative = (1.0 - weight) * a_grpo_scalar_expanded * negative_mask
    advantages = advantages_positive + advantages_negative
    
    # 6. 最终advantage
    advantages = advantages * eos_mask
    returns = token_level_return * eos_mask
    
    return advantages, returns

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_progress_advantage1(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   gamma: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    import time
    time0 = time.time()
    bsz, response_length = token_level_rewards.shape
        
    # ------------------------------------------------------------------
    # Step 1: 计算 G_token (时序累积回报) 和 G_episode (总回报)
    # ------------------------------------------------------------------
    # 对每个样本计算 G_token (时序累积回报)    
    token_level_return = torch.zeros_like(token_level_rewards)
    with torch.no_grad():
        
        # 1. 奖励掩码 (处理稀疏性和 EOS)
        # 只需要将填充部分的奖励设为 0，然后向量化函数会处理整个批次
        rewards_masked = token_level_rewards * eos_mask
        # 2. 调用向量化函数
        token_level_return = compute_discounted_return_vectorized(rewards_masked, gamma)
        # G_episode (总回报, G_0)
        g_episode = token_level_return[:, 0].clone()
            
    time1 = time.time()
    # ------------------------------------------------------------------
    # Step 2: 计算 A_GRPO_group (组优势 Agroup) (保持不变)
    # ------------------------------------------------------------------
    # Note: 计算 A_GRPO 所需的部分。
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(g_episode[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # 计算优势函数
        a_grpo_scalar = torch.zeros_like(g_episode)
        for i in range(bsz):
            idx = index[i]
            # (G_episode - mu) / sigma
            a_grpo_scalar[i] = (g_episode[i] - id2mean[idx]) / (id2std[idx] + epsilon)
    time2 = time.time()
    # ------------------------------------------------------------------
    # Step 3: 计算最终的加权优势 A_token (新方案 R)
    # ------------------------------------------------------------------    

    with torch.no_grad():
        
        # 1. 计算单样本范围 (Gmax - Gmin)
        # G_max,G_min: 找到每条轨迹所有 Gt 中的最大/最小值 (bs, 1)
        # G_range: (Gmax - Gmin) (bs, 1)
        G_max, _ = token_level_return.max(dim=-1, keepdim=True)
        G_min, _ = token_level_return.min(dim=-1, keepdim=True)
        G_range = G_max - G_min
        
        # 2. 扩展张量
        G0_base = g_episode.unsqueeze(-1).expand_as(token_level_return) # G0
        G_range_expanded = G_range.expand_as(token_level_return)        # Gmax - Gmin
        a_grpo_scalar_expanded = a_grpo_scalar.unsqueeze(-1).expand_as(token_level_return) # Agroup
        
        # 3. 计算局部相对优势 Delta_t'
        # 分子：G_t - G_0，分母：(Gmax - Gmin) + epsilon，局部相对优势 Delta_t'
        numerator = token_level_return - G0_base
        denominator = G_range_expanded + epsilon
        delta_t_prime = numerator / denominator
        
        # 4. 符号逻辑和修正因子 F_t
        # F_t = 1 + sign(A_GRPO) * Delta_t'
        sign_A_grpo = torch.sign(a_grpo_scalar_expanded)
        F_t = 1.0 + sign_A_grpo * delta_t_prime

    # 5. 最终优势 A_Final,t = F_t * A_GRPO_group
    advantages = F_t * a_grpo_scalar_expanded
    
    # 6. 应用最终的 EOS 掩码
    advantages = advantages * eos_mask
    returns = token_level_return * eos_mask
    
    return advantages, returns

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_progress_advantage2(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   gamma: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    import time
    time0 = time.time()
    bsz, response_length = token_level_rewards.shape
        
    # ------------------------------------------------------------------
    # Step 1: 计算 G_token (时序累积回报) 和 G_episode (总回报)
    # ------------------------------------------------------------------
    # 对每个样本计算 G_token (时序累积回报)    
    token_level_return = torch.zeros_like(token_level_rewards)
    with torch.no_grad():
        
        # 1. 奖励掩码 (处理稀疏性和 EOS)
        # 只需要将填充部分的奖励设为 0，然后向量化函数会处理整个批次
        rewards_masked = token_level_rewards * eos_mask
        # 2. 调用向量化函数
        token_level_return = compute_discounted_return_vectorized(rewards_masked, gamma)
        # G_episode (总回报, G_0)
        g_episode = token_level_return[:, 0].clone()
            
    time1 = time.time()
    # ------------------------------------------------------------------
    # Step 2: 计算 A_GRPO_group (组优势 Agroup) (保持不变)
    # ------------------------------------------------------------------
    # Note: 计算 A_GRPO 所需的部分。
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(g_episode[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # 计算优势函数
        a_grpo_scalar = torch.zeros_like(g_episode)
        for i in range(bsz):
            idx = index[i]
            # (G_episode - mu) / sigma
            a_grpo_scalar[i] = (g_episode[i] - id2mean[idx]) / (id2std[idx] + epsilon)
    time2 = time.time()
    # ------------------------------------------------------------------
    # Step 3: 计算最终的加权优势 A_token
    # ------------------------------------------------------------------
    # A_token = (G_token / G_episode) * A_GRPO_group
    g_episode_tiled = g_episode.unsqueeze(-1).expand_as(token_level_return)
    g_episode_abs = torch.abs(g_episode_tiled)
    weight_factor = (token_level_return / (g_episode_abs + epsilon)) * eos_mask
    weight_factor = torch.clamp(weight_factor, max=1.0)
    # weight_factor = (token_level_return / (g_episode_tiled + epsilon)) * eos_mask
    # weight_factor = torch.clamp(weight_factor, max=1.0)
    a_grpo_tiled = a_grpo_scalar.unsqueeze(-1).expand_as(token_level_return)
    advantages = weight_factor * a_grpo_tiled * eos_mask
    returns = g_episode_tiled * eos_mask
    time3 = time.time()
    # print("time1", time1-time0)
    # print("time2", time2-time1)
    # print("time3", time3-time2)
    return advantages, returns

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_progress_advantage3(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   gamma: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    import time
    time0 = time.time()
    bsz, response_length = token_level_rewards.shape
        
    # ------------------------------------------------------------------
    # Step 1: 计算 G_token (时序累积回报) 和 G_episode (总回报)
    # ------------------------------------------------------------------
    # 对每个样本计算 G_token (时序累积回报)    
    token_level_return = torch.zeros_like(token_level_rewards)
    with torch.no_grad():
        
        # 1. 奖励掩码 (处理稀疏性和 EOS)
        # 只需要将填充部分的奖励设为 0，然后向量化函数会处理整个批次
        rewards_masked = token_level_rewards * eos_mask
        # 2. 调用向量化函数
        token_level_return = compute_discounted_return_vectorized(rewards_masked, gamma)
        # G_episode (总回报, G_0)
        g_episode = token_level_return[:, 0].clone()
            
    time1 = time.time()
    # ------------------------------------------------------------------
    # Step 2: 计算 A_GRPO_group (组优势 Agroup) (保持不变)
    # ------------------------------------------------------------------
    # Note: 计算 A_GRPO 所需的部分。
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(g_episode[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # 计算优势函数
        a_grpo_scalar = torch.zeros_like(g_episode)
        for i in range(bsz):
            idx = index[i]
            # (G_episode - mu) / sigma
            a_grpo_scalar[i] = (g_episode[i] - id2mean[idx]) / (id2std[idx] + epsilon)
    time2 = time.time()
    # ------------------------------------------------------------------
    # Step 3: 计算最终的加权优势 A_token
    # ------------------------------------------------------------------
    # A_token = (G_token / G_episode) * A_GRPO_group
    g_episode_tiled = g_episode.unsqueeze(-1).expand_as(token_level_return)
    g_episode_abs = torch.abs(g_episode_tiled)
    weight_factor = (torch.abs(token_level_return) / (g_episode_abs + epsilon)) * eos_mask
    weight_factor = torch.clamp(weight_factor, max=1.0)
    # weight_factor = (token_level_return / (g_episode_tiled + epsilon)) * eos_mask
    # weight_factor = torch.clamp(weight_factor, max=1.0)
    a_grpo_tiled = a_grpo_scalar.unsqueeze(-1).expand_as(token_level_return)
    advantages = weight_factor * a_grpo_tiled * eos_mask
    returns = g_episode_tiled * eos_mask
    time3 = time.time()
    # print("time1", time1-time0)
    # print("time2", time2-time1)
    # print("time3", time3-time2)
    return advantages, returns

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_progress_advantage4(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   gamma: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    import time
    time0 = time.time()
    bsz, response_length = token_level_rewards.shape
        
    # ------------------------------------------------------------------
    # Step 1: 计算 G_token (时序累积回报) 和 G_episode (总回报)
    # ------------------------------------------------------------------
    # 对每个样本计算 G_token (时序累积回报)    
    token_level_return = torch.zeros_like(token_level_rewards)
    with torch.no_grad():
        
        # 1. 奖励掩码 (处理稀疏性和 EOS)
        # 只需要将填充部分的奖励设为 0，然后向量化函数会处理整个批次
        rewards_masked = token_level_rewards * eos_mask
        # 2. 调用向量化函数
        token_level_return = compute_discounted_return_vectorized(rewards_masked, gamma)
        # G_episode (总回报, G_0)
        g_episode = token_level_return[:, 0].clone()
            
    time1 = time.time()
    # ------------------------------------------------------------------
    # Step 2: 计算 A_GRPO_group (组优势 Agroup) (保持不变)
    # ------------------------------------------------------------------
    # Note: 计算 A_GRPO 所需的部分。
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(g_episode[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # 计算优势函数
        a_grpo_scalar = torch.zeros_like(g_episode)
        for i in range(bsz):
            idx = index[i]
            # (G_episode - mu) / sigma
            a_grpo_scalar[i] = (g_episode[i] - id2mean[idx]) / (id2std[idx] + epsilon)
    time2 = time.time()
    # ------------------------------------------------------------------
    # Step 3: 计算最终的加权优势 A_token
    # ------------------------------------------------------------------
    # A_token = (G_token / G_episode) * A_GRPO_group
    g_episode_tiled = g_episode.unsqueeze(-1).expand_as(token_level_return)
    weight_factor = (token_level_return / (g_episode_tiled + epsilon)) * eos_mask
    weight_factor = torch.clamp(weight_factor, max=1.0)
    a_grpo_tiled = a_grpo_scalar.unsqueeze(-1).expand_as(token_level_return)
    sign_A_grpo = torch.sign(g_episode_tiled*a_grpo_tiled)
    advantages = sign_A_grpo * weight_factor * a_grpo_tiled * eos_mask
    returns = g_episode_tiled * eos_mask
    time3 = time.time()
    return advantages, returns

def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_progress_advantage5(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   gamma: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    import time
    time0 = time.time()
    bsz, response_length = token_level_rewards.shape
        
    # ------------------------------------------------------------------
    # Step 1: 计算 G_token (时序累积回报) 和 G_episode (总回报)
    # ------------------------------------------------------------------
    # 对每个样本计算 G_token (时序累积回报)    
    token_level_return = torch.zeros_like(token_level_rewards)
    with torch.no_grad():
        
        # 1. 奖励掩码 (处理稀疏性和 EOS)
        # 只需要将填充部分的奖励设为 0，然后向量化函数会处理整个批次
        rewards_masked = token_level_rewards * eos_mask
        # 2. 调用向量化函数
        token_level_return = compute_discounted_return_vectorized(rewards_masked, gamma)
        # G_episode (总回报, G_0)
        g_episode = token_level_return[:, 0].clone()
            
    time1 = time.time()
    # ------------------------------------------------------------------
    # Step 2: 计算 A_GRPO_group (组优势 Agroup) (保持不变)
    # ------------------------------------------------------------------
    # Note: 计算 A_GRPO 所需的部分。
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(g_episode[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # 计算优势函数
        a_grpo_scalar = torch.zeros_like(g_episode)
        for i in range(bsz):
            idx = index[i]
            # (G_episode - mu) / sigma
            a_grpo_scalar[i] = (g_episode[i] - id2mean[idx]) / (id2std[idx] + epsilon)
    time2 = time.time()
    # ------------------------------------------------------------------
    # Step 3: 计算最终的加权优势 A_token (新方案 R)
    # ------------------------------------------------------------------    

    with torch.no_grad():
        
        # 1. 计算单样本范围 (Gmax - Gmin)
        # G_max,G_min: 找到每条轨迹所有 Gt 中的最大/最小值 (bs, 1)
        # G_range: (Gmax - Gmin) (bs, 1)
        G_max, _ = token_level_return.max(dim=-1, keepdim=True)
        G_min, _ = token_level_return.min(dim=-1, keepdim=True)
        G_range = G_max - G_min
        
        # 2. 扩展张量
        G0_base = g_episode.unsqueeze(-1).expand_as(token_level_return) # G0
        G_range_expanded = G_range.expand_as(token_level_return)        # Gmax - Gmin
        a_grpo_scalar_expanded = a_grpo_scalar.unsqueeze(-1).expand_as(token_level_return) # Agroup
        
        # 3. 计算局部相对优势 Delta_t'
        # 分子：G_t - G_0，分母：(Gmax - Gmin) + epsilon，局部相对优势 Delta_t'
        numerator = token_level_return - G0_base
        denominator = G_range_expanded + epsilon
        delta_t_prime = numerator / denominator
        delta_t_prime = torch.clamp(delta_t_prime, min=-1.0, max=1.0)
        
        # 4. 符号逻辑和修正因子 F_t
        # F_t = 1 + sign(A_GRPO) * Delta_t'
        sign_A_grpo = torch.sign(a_grpo_scalar_expanded)
        F_t = 1.0 + 0.2*sign_A_grpo * delta_t_prime

    # 5. 最终优势 A_Final,t = F_t * A_GRPO_group
    advantages = F_t * a_grpo_scalar_expanded
    
    # 6. 应用最终的 EOS 掩码
    advantages = advantages * eos_mask
    returns = token_level_return * eos_mask
    
    return advantages, returns

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_progress_advantage6(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   gamma: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    import time
    time0 = time.time()
    bsz, response_length = token_level_rewards.shape
        
    # ------------------------------------------------------------------
    # Step 1: 计算 G_token (时序累积回报) 和 G_episode (总回报)
    # ------------------------------------------------------------------
    token_level_return = torch.zeros_like(token_level_rewards)
    with torch.no_grad():
        
        # 1. 奖励掩码 (处理稀疏性和 EOS)
        # 只需要将填充部分的奖励设为 0，然后向量化函数会处理整个批次
        rewards_masked = token_level_rewards * eos_mask
        
        # 2. 调用向量化函数
        token_level_return = compute_discounted_return_vectorized(rewards_masked, gamma)
        
        # G_episode (总回报, G_0)
        g_episode = token_level_return[:, 0].clone()
            
    time1 = time.time()
    # ------------------------------------------------------------------
    # Step 2: 计算 A_GRPO_group (组优势)
    # ------------------------------------------------------------------
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(g_episode[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # 计算优势函数
        a_grpo_scalar = torch.zeros_like(g_episode)
        for i in range(bsz):
            idx = index[i]
            # (G_episode - mu) / sigma
            a_grpo_scalar[i] = (g_episode[i] - id2mean[idx]) / (id2std[idx] + epsilon)
    time2 = time.time()
    # ------------------------------------------------------------------
    # Step 3: 计算最终的加权优势 A_token
    # ------------------------------------------------------------------
    with torch.no_grad():
        # 1. 找到 Gt 集合的实际最小值 Gt_t_min = min(Gt)
        # Gt_t_min_per_seq: shape (bs, 1)
        G_max, _ = token_level_return.max(dim=-1, keepdim=True)
        G_min, _ = token_level_return.min(dim=-1, keepdim=True)
        G_range = G_max - G_min
        
        # 将标量 (G0) 和 Gmin 扩展为 (bs, response_length)
        G_tiled = g_episode.unsqueeze(-1).expand_as(token_level_return)
        
        # 3. 计算 Weight: weight = clamp((Gt - Gmin) / (G - Gmin), 0, 1)
        # weight: (Gt - Gmin) / (G0 - Gmin); Clamp: clamp(..., 0, 1)
        denominator = G_range + epsilon
        weight_factor = (token_level_return - G_min) / denominator
        weight = torch.clamp(weight_factor, min=0.0, max=1.0)
        
        # 4. 准备 GRPO 优势的平铺版本和掩码
        # 创建用于分段计算的掩码
        a_grpo_scalar_expanded = a_grpo_scalar.unsqueeze(-1).expand_as(token_level_return)
        positive_mask = (a_grpo_scalar_expanded >= 0).float()
        negative_mask = 1.0 - positive_mask
    
        # 5. 分段计算 advantages
        # A_t = weight * A_GRPO_group  (if A_GRPO >= 0)
        # A_t = (1 - weight) * A_GRPO_group  (if A_GRPO < 0)
        advantages_positive = weight * a_grpo_scalar_expanded * positive_mask
        advantages_negative = (1.0 - weight) * a_grpo_scalar_expanded * negative_mask
        advantages = advantages_positive + advantages_negative
    
    # 6. 最终advantage
    advantages = advantages * eos_mask
    returns = token_level_return * eos_mask
    
    return advantages, returns

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_progress_advantage7(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   gamma: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    import time
    time0 = time.time()
    bsz, response_length = token_level_rewards.shape
        
    # ------------------------------------------------------------------
    # Step 1: 计算 G_token (时序累积回报) 和 G_episode (总回报)
    # ------------------------------------------------------------------
    # 对每个样本计算 G_token (时序累积回报)    
    token_level_return = torch.zeros_like(token_level_rewards)
    with torch.no_grad():
        
        # 1. 奖励掩码 (处理稀疏性和 EOS)
        # 只需要将填充部分的奖励设为 0，然后向量化函数会处理整个批次
        rewards_masked = token_level_rewards * eos_mask
        # 2. 调用向量化函数
        token_level_return, token_level_return_sparse = compute_discounted_return_iterative_sparse(rewards_masked, gamma)
        # G_episode (总回报, G_0)
        g_episode = token_level_return[:, 0].clone()
            
    time1 = time.time()
    # ------------------------------------------------------------------
    # Step 2: 计算 A_GRPO_group (组优势 Agroup) (保持不变)
    # ------------------------------------------------------------------
    # Note: 计算 A_GRPO 所需的部分。
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(g_episode[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # 计算优势函数
        a_grpo_scalar = torch.zeros_like(g_episode)
        for i in range(bsz):
            idx = index[i]
            # (G_episode - mu) / sigma
            a_grpo_scalar[i] = (g_episode[i] - id2mean[idx]) / (id2std[idx] + epsilon)
    time2 = time.time()
    # ------------------------------------------------------------------
    # Step 3: 计算最终的加权优势 A_token (新方案 R)
    # ------------------------------------------------------------------    

    with torch.no_grad():
        
        # 1. 计算每个样本中非零 G_t 的数量 N_sparse
        # 使用 token_level_return_sparse != 0 来确保只计算实际有回报的位置
        N_sparse = (token_level_return_sparse != 0).sum(dim=-1, keepdim=True)
        # 防止除以零，将 N_sparse <= 0 的位置设为 1
        N_sparse = torch.where(N_sparse == 0, torch.tensor(1.0, device=N_sparse.device), N_sparse.float())
        
        # 2. 计算稀疏回报的均值 G_sparse_mean (bs, 1)
        # 由于 token_level_return_sparse 已经在 R_t=0 的位置为 0，可以直接对它求和
        G_sparse_sum = token_level_return_sparse.sum(dim=-1, keepdim=True)
        G_sparse_mean = G_sparse_sum / N_sparse
        
        # 1. 计算单样本范围 (Gmax - Gmin)
        # G_max,G_min: 找到每条轨迹所有 Gt 中的最大/最小值 (bs, 1)
        # G_range: (Gmax - Gmin) (bs, 1)
        G_max, _ = token_level_return.max(dim=-1, keepdim=True)
        G_min, _ = token_level_return.min(dim=-1, keepdim=True)
        G_range = G_max - G_min
        
        # 2. 扩展张量
        G_range_expanded = G_range.expand_as(token_level_return)        # Gmax - Gmin
        a_grpo_scalar_expanded = a_grpo_scalar.unsqueeze(-1).expand_as(token_level_return) # Agroup
        
        # 3. 计算局部相对优势 Delta_t'
        # 分子：G_t - G_0，分母：(Gmax - Gmin) + epsilon，局部相对优势 Delta_t'
        numerator = token_level_return - G_sparse_mean
        denominator = G_range_expanded + epsilon
        delta_t_prime = numerator / denominator
        delta_t_prime = torch.clamp(delta_t_prime, min=-1.0, max=1.0)
        
        # 4. 符号逻辑和修正因子 F_t
        # F_t = 1 + sign(A_GRPO) * Delta_t'
        sign_A_grpo = torch.sign(a_grpo_scalar_expanded)
        F_t = 1.0 + 0.5*sign_A_grpo * delta_t_prime

    # 5. 最终优势 A_Final,t = F_t * A_GRPO_group
    advantages = F_t * a_grpo_scalar_expanded
    
    # 6. 应用最终的 EOS 掩码
    advantages = advantages * eos_mask
    returns = token_level_return * eos_mask
    
    return advantages, returns

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_progress_advantage8(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   gamma: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    import time
    time0 = time.time()
    bsz, response_length = token_level_rewards.shape
    device = token_level_rewards.device
        
    # ------------------------------------------------------------------
    # Step 1: 计算 G_token (时序累积回报) 和 G_episode (总回报)
    # ------------------------------------------------------------------
    # 对每个样本计算 G_token (时序累积回报)    
    token_level_return = torch.zeros_like(token_level_rewards)
    with torch.no_grad():
        
        # 1. 奖励掩码 (处理稀疏性和 EOS)
        # 只需要将填充部分的奖励设为 0，然后向量化函数会处理整个批次
        rewards_masked = token_level_rewards * eos_mask
        # 2. 调用向量化函数
        token_level_return, token_level_return_sparse = compute_discounted_return_iterative_sparse(rewards_masked, gamma)
        # G_episode (总回报, G_0)
        g_episode = token_level_return[:, 0].clone()
            
    time1 = time.time()
    # ------------------------------------------------------------------
    # Step 2: 提取、对齐关键回报，并建立映射
    # ------------------------------------------------------------------
    with torch.no_grad():
        # 1. 提取非零回报的索引和值
        sparse_mask = (token_level_return_sparse != 0)
        N_sparse = sparse_mask.sum(dim=-1) # 每个样本的非零回报数量 (bsz,)
        L_max_sparse = N_sparse.max().item() # 最长非零回报长度
        
        if L_max_sparse == 0:
            return torch.zeros_like(token_level_rewards), token_level_return * eos_mask

        G_sparse_aligned = torch.zeros(bsz, L_max_sparse, device=device)
        # sparse_index_map 存储原始 t 对应的对齐后的 k 索引
        sparse_index_map = -torch.ones_like(token_level_rewards, dtype=torch.long, device=device) 

        # 2. 提取、填充并建立映射
        for i in range(bsz):
            if N_sparse[i] > 0:
                valid_indices = sparse_mask[i].nonzero(as_tuple=True)[0] # 原始 t 索引
                g_sparse_values = token_level_return_sparse[i, valid_indices]
                
                # 填充到 G_sparse_aligned
                G_sparse_aligned[i, :N_sparse[i]] = g_sparse_values
                
                # 建立映射：sparse_index_map[i, t] = k
                for k_idx, t_idx in enumerate(valid_indices):
                    sparse_index_map[i, t_idx] = k_idx
                
                # 填充剩余部分：用最后一个有效值填充
                last_val = g_sparse_values[-1]
                G_sparse_aligned[i, N_sparse[i]:] = last_val
    
    # ------------------------------------------------------------------
    # Step 3: 对齐后的关键回报应用 GRPO (时序 GRPO)
    # ------------------------------------------------------------------
    advantages_sparse_aligned = torch.zeros_like(G_sparse_aligned)
    index_expanded = index.unsqueeze(-1).expand(-1, L_max_sparse) 
    
    for k in range(L_max_sparse):
        G_k = G_sparse_aligned[:, k] # 所有样本在关键点 k 上的回报 (bsz,)
        Index_k = index_expanded[:, k] 
        
        # ⚠️ 关键修正：只对实际有数据的样本计算统计量
        N_sparse_at_k = (N_sparse.squeeze(-1) >= k + 1) # 长度 >= k+1 的样本才算作有真实数据
        
        G_k_valid = G_k[N_sparse_at_k]
        Index_k_valid = Index_k[N_sparse_at_k]
        
        # 如果当前关键点 k 没有任何样本有真实回报（理论上不应发生，但作为防护）
        if G_k_valid.numel() == 0:
            continue

        # 1. 计算每个 prompt group 的统计量 (μ_k, σ_k)
        id2score_k = defaultdict(list)
        for i in range(G_k_valid.shape[0]):
            id2score_k[Index_k_valid[i].item()].append(G_k_valid[i])
            
        # 2. 计算优势函数 A_k
        A_k_full = torch.zeros_like(G_k) # 优势计算结果 (bsz,)
        
        # 遍历有效样本，计算它们的优势
        for i in range(G_k_valid.shape[0]):
            original_idx = torch.where(N_sparse_at_k)[0][i].item() # 原始样本索引
            idx = Index_k_valid[i].item()
            scores_tensor = torch.stack(id2score_k[idx])
            
            if len(scores_tensor) <= 1:
                mu_k = torch.tensor(0.0, device=device)
                std_k = torch.tensor(1.0, device=device)
            else:
                mu_k = torch.mean(scores_tensor)
                std_k = torch.std(scores_tensor)
            
            # 计算优势 (G_k - mu_k) / sigma_k
            A_k_full[original_idx] = (G_k_valid[i] - mu_k) / (std_k + epsilon)
            
        # 3. 存储结果
        advantages_sparse_aligned[:, k] = A_k_full

    # ------------------------------------------------------------------
    # Step 4: 优势映射回原始长度 (Advantages Mapping and Sharing)
    # ------------------------------------------------------------------
    advantages = torch.zeros_like(token_level_rewards)
    
    # 使用向量化的前向填充逻辑
    for i in range(bsz):
        temp_map = sparse_index_map[i].clone()
        
        # 1. 将 -1 (R_t=0 的 Token) 替换为一个很小的负数，然后使用 cummax
        #   - 目的：找到每个 R_t=0 Token 之前的最近的 R_t'!=0 Token 的 k_idx
        temp_map[temp_map == -1] = -100 
        filled_map = torch.cummax(temp_map, dim=0).values.long()
        
        # 2. 找到需要填充/映射的位置
        
        # R_t != 0 的位置： k_idx 就是 sparse_index_map[i, t]
        valid_mask = (sparse_index_map[i] != -1)
        
        # R_t = 0 的位置： k_idx 是 cummax 填充的 filled_map
        fill_mask = (sparse_index_map[i] == -1)
        
        # 3. 映射优势
        # 映射 R_t != 0 的位置
        advantages[i, valid_mask] = advantages_sparse_aligned[i, sparse_index_map[i, valid_mask].long()]

        # 映射 R_t = 0 的位置 (共享优势)
        if fill_mask.any():
            k_indices_to_fill = filled_map[fill_mask]
            
            # 钳位，防止 cummax 产生的负数（如起始位置）越界
            k_indices_to_fill = torch.clamp(k_indices_to_fill, 0, L_max_sparse - 1)
            
            # 填充
            advantages[i, fill_mask] = advantages_sparse_aligned[i, k_indices_to_fill]

    # ------------------------------------------------------------------
    # Step 5: 最终掩码
    # ------------------------------------------------------------------
    advantages = advantages * eos_mask
    returns = token_level_return * eos_mask
    
    return advantages, returns

def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
