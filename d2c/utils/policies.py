"""Policies used by various agents."""

import torch
import numpy as np
from torch import nn, Tensor
from typing import Union
from d2c.utils.utils import to_array_as


class DeterministicPolicy(nn.Module):
    """Returns deterministic action."""

    def __init__(self, a_network: nn.Module) -> None:
        super(DeterministicPolicy, self).__init__()
        self._a_network = a_network

    def forward(self, observation: Union[np.ndarray, Tensor]) -> np.ndarray:
        with torch.no_grad():
            action = self._a_network(observation)
        return to_array_as(action, observation)


class DeterministicSoftPolicy(nn.Module):
    """Returns mode of policy distribution."""

    def __init__(self, a_network: nn.Module) -> None:
        super(DeterministicSoftPolicy, self).__init__()
        self._a_network = a_network

    def forward(self, observation: Union[np.ndarray, Tensor]) -> np.ndarray:
        with torch.no_grad():
            action = self._a_network(observation)[0]
        return to_array_as(action, observation)


class IDQLArgmaxPolicy(nn.Module):
    """
    IDQL Argmax Policy: Samples N actions from the Diffusion Actor (p_fn)
    and selects the action with the maximum Q-value (q_fn).

    Args:
        p_fn (nn.Module): The Diffusion Actor network.
        q_fn (nn.Module): The minimum Q-network (Critic).
        num_samples (int): Number of candidate actions to sample per state.
        device (str): The device for tensor operations.
    """
    def __init__(
            self,
            p_fn: nn.Module,
            q_fn: nn.Module,
            num_samples: int = 64,
            device: str = 'cpu'
    ) -> None:
        super(IDQLArgmaxPolicy, self).__init__()
        self._p_fn = p_fn
        self._q_fn = q_fn
        self.num_samples = num_samples
        self.device = device

    def forward(self, observation: Union[np.ndarray, Tensor]) -> np.ndarray:
        is_numpy = isinstance(observation, np.ndarray)

        if is_numpy:
            state = torch.tensor(observation, dtype=torch.float32, device=self.device)
        else:
            state = observation.to(self.device)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        batch_size, state_dim = state.shape

        with torch.no_grad():
            # 1. 采样 N 个候选动作: (Batch, N, Action_Dim)
            # (注意: p_fn 必须实现了 sample_n 方法)
            actions = self._p_fn.sample_n(state, n=self.num_samples)
            action_dim = actions.shape[-1]

            # 2. 准备 Q 函数的输入数据 (展平操作)
            # State 扩展: (B, D) -> (B*N, D)
            state_repeated = (
                state.unsqueeze(1)
                .repeat(1, self.num_samples, 1)
                .view(-1, state_dim)
            )

            # Actions 展平: (B, N, D) -> (B*N, D)
            actions_flattened = actions.view(-1, action_dim)

            # 3. 评估 Q 值: q_values_flat shape: (B*N, )
            q_values_flat = self._q_fn(state_repeated, actions_flattened)

            # 4. 选择 Q 值最高的动作
            # 重塑回 (Batch, N) 并在 N 维度上找最大值
            q_values = q_values_flat.view(batch_size, self.num_samples)
            best_indices = q_values.argmax(dim=1) # (Batch,)

            # 5. 提取最优动作: (Batch, Action_Dim)
            # 使用高级索引进行提取
            best_actions = actions[torch.arange(batch_size), best_indices].squeeze()

        # 6. 后处理: 将 Tensor 转换为 NumPy Array
        # 使用 to_array_as 确保输出格式与用户环境要求一致
        return to_array_as(best_actions, observation)