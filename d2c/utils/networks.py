"""Neural networks for RL models."""
import math

import torch
import numpy as np
import torch.nn.functional as F
from gym.spaces import Box, Space
from torch import nn, Tensor
from typing import Tuple, List, Union, Type, Optional, Sequence
from torch.distributions import Normal, TransformedDistribution, Distribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform

ModuleType = Type[nn.Module]
LOG_STD_MIN = -5
LOG_STD_MAX = 2


def miniblock(
        input_size: int,
        output_size: int = 0,
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers


def get_spec_means_mags(
        space: Box,
        device: Optional[Union[str, int, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    means = (space.high + space.low) / 2.0
    mags = (space.high - space.low) / 2.0
    means = torch.as_tensor(means, device=device, dtype=torch.float32)
    mags = torch.as_tensor(mags, device=device, dtype=torch.float32)
    return means, mags


class ActorNetwork(nn.Module):
    """Stochastic Actor network.

    :param Box observation_space: the observation space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param Box action_space: the action space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(ActorNetwork, self).__init__()
        self._device = device
        state_dim = observation_space.shape[0]
        self._action_space = action_space
        self._action_dim = action_space.shape[0]
        self._layers = []
        hidden_sizes = [state_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        output_dim = self._action_dim * 2
        self._layers += [nn.Linear(hidden_sizes[-1], output_dim)]
        self._model = nn.Sequential(*self._layers)
        self._action_means, self._action_mags = get_spec_means_mags(
            self._action_space, self._device)

    def _get_output(self, state: Union[np.ndarray, torch.Tensor]) \
            -> Tuple[Distribution, torch.Tensor]:
        state = torch.as_tensor(state, device=self._device, dtype=torch.float32)
        h = self._model(state)
        mean, log_std = torch.split(h, split_size_or_sections=[self._action_dim, self._action_dim], dim=-1)
        a_tanh_mode = torch.tanh(mean) * self._action_mags + self._action_means
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)
        # TanhTransform()is equivalent to \
        # ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
        a_distribution = TransformedDistribution(
            base_distribution=Normal(
                loc=mean,
                scale=std,
            ),
            transforms=[
                AffineTransform(0., 2.),
                SigmoidTransform(),
                AffineTransform(-1., 2.),
                AffineTransform(loc=self._action_means, scale=self._action_mags)
            ],
        )
        return a_distribution, a_tanh_mode

    def forward(self, state: Union[np.ndarray, torch.Tensor])\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_dist, a_tanh_mode = self._get_output(state)
        a_sample = a_dist.rsample()
        log_pi_a = a_dist.log_prob(a_sample)
        return a_tanh_mode, a_sample, log_pi_a

    def get_log_density(self, state: Tensor, action: Tensor) -> Tensor:
        a_dist, _ = self._get_output(state)
        action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        log_density = a_dist.log_prob(action)
        return log_density

    def sample_n(self, state: Union[np.ndarray, Tensor], n: int = 1)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_dist, a_tanh_mode = self._get_output(state)
        a_sample = a_dist.rsample([n])
        log_pi_a = a_dist.log_prob(a_sample)
        return a_tanh_mode, a_sample, log_pi_a

    def sample(self, state: Union[np.ndarray, Tensor]) -> Tensor:
        return self.sample_n(state, n=1)[1][0]

    @property
    def action_space(self) -> Box:
        return self._action_space


class ActorNetworkDet(nn.Module):
    """Deterministic Actor network.

    :param Box observation_space: the observation space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param Box action_space: the action space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(ActorNetworkDet, self).__init__()
        self._device = device
        state_dim = observation_space.shape[0]
        self._action_space = action_space
        self._action_dim = action_space.shape[0]
        self._layers = []
        hidden_sizes = [state_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], self._action_dim)]
        self._model = nn.Sequential(*self._layers)
        self._action_means, self._action_mags = get_spec_means_mags(
            self._action_space, self._device)

    def forward(self, state: Union[np.ndarray, Tensor]) -> Tensor:
        state = torch.as_tensor(state, device=self._device, dtype=torch.float32)
        a = self._model(state)
        return torch.tanh(a) * self._action_mags + self._action_means

    @property
    def action_space(self) -> Box:
        return self._action_space


class ProbDynamicsNetwork(nn.Module):
    """Stochastic Dynamics network(Probabilistic dynamics model).

    :param int state_dim: the observation space dimension.
    :param int action_dim: the action space dimension.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param bool local_mode: `local_mode` means that this model predicts the difference to the current state.
    :param bool with_reward: if the output of the dynamics contains the reward or not.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            fc_layer_params: Sequence[int] = (),
            local_mode: bool = False,
            with_reward: bool = False,
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(ProbDynamicsNetwork, self).__init__()
        self._local_mode = local_mode
        self._with_reward = with_reward
        self._device = device
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._layers = []
        hidden_sizes = [state_dim + self._action_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        output_dim = (self._state_dim + with_reward) * 2
        self._layers += [nn.Linear(hidden_sizes[-1], output_dim)]
        self._model = nn.Sequential(*self._layers)
        # logstd bounds
        init_max = torch.empty(1, state_dim + with_reward, dtype=torch.float32).fill_(2.0)
        init_min = torch.empty(1, state_dim + with_reward, dtype=torch.float32).fill_(-10.0)
        self._max_logstd = nn.Parameter(init_max)
        self._min_logstd = nn.Parameter(init_min)

    def _get_output(
            self,
            state: Union[np.ndarray, torch.Tensor],
            action: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[Distribution, torch.Tensor]:
        state = torch.as_tensor(state, device=self._device, dtype=torch.float32)
        action = torch.as_tensor(action, device=self._device, dtype=torch.float32)
        h = torch.cat([state, action], 1)
        h = self._model(h)
        mean, log_std = torch.split(h, split_size_or_sections=self._state_dim + self._with_reward, dim=-1)
        log_std = self._max_logstd - F.softplus(self._max_logstd - log_std)
        log_std = self._min_logstd + F.softplus(log_std - self._min_logstd)
        std = torch.exp(log_std)
        if self._local_mode:
            if self._with_reward:
                s_p, reward = torch.split(mean, [self._state_dim, 1], dim=-1)
                s_p = s_p + state
                mean = torch.cat([s_p, reward], -1)
            else:
                mean = mean + state
        dist = Normal(loc=mean, scale=std)
        return dist, mean

    def forward(
            self,
            state: Union[np.ndarray, torch.Tensor],
            action: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Distribution]:
        dist, mean = self._get_output(state, action)
        sample = dist.rsample()
        return mean, sample, dist

    def get_log_density(
            self,
            state: Tensor,
            action: Tensor,
            output: Tensor
    ) -> Tensor:
        assert output.shape[-1] == self._state_dim + self._with_reward, 'Wrong target dimension!'
        dist, _ = self._get_output(state, action)
        output = torch.as_tensor(output, dtype=torch.float32, device=self._device)
        log_density = dist.log_prob(output)
        return log_density

    @property
    def max_logstd(self):
        return self._max_logstd

    @property
    def min_logstd(self):
        return self._min_logstd


class CriticNetwork(nn.Module):
    """Critic Network.

    :param gym.spaces.Box or int observation_space: the observation space information. It is an instance
        of class: ``gym.spaces.Box``. `observation_space` can also be an integer which
        represents the dimension of the observation.
    :param gym.spaces.Box or int action_space: the action space information. It is an instance
        of class: ``gym.spaces.Box``. `action_space` can also be an integer which
        represents the dimension of the action.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            observation_space: Union[Box, Space, int],
            action_space: Union[Box, Space, int],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(CriticNetwork, self).__init__()
        self._device = device
        if isinstance(observation_space, int):
            state_dim = observation_space
        else:
            state_dim = observation_space.shape[0]
        if isinstance(action_space, int):
            action_dim = action_space
        else:
            action_dim = action_space.shape[0]
        self._layers = []
        hidden_sizes = [state_dim + action_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], 1)]
        self._model = nn.Sequential(*self._layers)

    def forward(
            self,
            state: Union[np.ndarray, Tensor],
            action: Union[np.ndarray, Tensor]
    ) -> Tensor:
        state = torch.as_tensor(state, device=self._device, dtype=torch.float32)
        action = torch.as_tensor(action, device=self._device, dtype=torch.float32)
        h = torch.cat([state, action], dim=-1)
        h = self._model(h)
        return torch.reshape(h, [-1])


class MLP(nn.Module):
    """Multi-layer Perceptron.

    :param int input_dim: the dimension of the input.
    :param int output_dim: the dimension of the output.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(MLP, self).__init__()
        self._device = device
        self._layers = []
        hidden_sizes = [input_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], output_dim)]
        self._model = nn.Sequential(*self._layers)

    def forward(self, inputs: Union[np.ndarray, Tensor]) -> Tensor:
        inputs = torch.as_tensor(inputs, device=self._device, dtype=torch.float32)
        return self._model(inputs)


class Classifier(nn.Module):
    """ based on Multi-layer Perceptron. Discriminator network for H2O.

    :param int input_dim: the dimension of the input.
    :param int output_dim: the dimension of the output.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int = 2,
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(Classifier, self).__init__()
        self._device = device
        self._layers = []
        hidden_sizes = [input_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += miniblock(hidden_sizes[-1], output_dim, None, nn.Tanh)
        self._model = nn.Sequential(*self._layers)

    def forward(self, inputs: Union[np.ndarray, Tensor]) -> Tensor:
        inputs = torch.as_tensor(inputs, device=self._device, dtype=torch.float32)
        return self._model(inputs) * 2


class ConcatClassifier(Classifier):
    """Concatenate inputs along dimension and then pass through MLP.

    :param int dim: concatenate inputs in row or column (0 or 1).
    """

    def __init__(
            self,
            dim: int = 1,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs: Union[np.ndarray, Tensor]) -> Tensor:
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs)


class Scalar(nn.Module):
    """ Scalar network

    :param float init_value: initialized value for the scalar
    """

    def __init__(
            self,
            init_value: float,
            device: Union[str, int, torch.device] = 'cpu'
    ) -> None:
        super().__init__()
        self._device = device
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32).to(self._device)
        )

    def forward(self) -> Tensor:
        return self.constant


class Discriminator(nn.Module):
    """A Discriminator Network(for DMIL).

    :param Box observation_space: the observation space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param Box action_space: the action space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(Discriminator, self).__init__()
        self._device = device
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self._layers = []
        hidden_sizes = [2 * state_dim + 2 * action_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += miniblock(hidden_sizes[-1], 1, None, nn.Sigmoid)
        self._model = nn.Sequential(*self._layers)

    def forward(
            self,
            state: Union[np.ndarray, Tensor],
            action: Union[np.ndarray, Tensor],
            logpi: Union[np.ndarray, Tensor],
            lossf: Union[np.ndarray, Tensor]
    ) -> Tensor:
        state = torch.as_tensor(state, device=self._device, dtype=torch.float32)
        action = torch.as_tensor(action, device=self._device, dtype=torch.float32)
        logpi = torch.as_tensor(logpi, device=self._device, dtype=torch.float32)
        lossf = torch.as_tensor(lossf, device=self._device, dtype=torch.float32)
        h = torch.cat([state, action, logpi, lossf], dim=-1)
        h = self._model(h)
        return torch.reshape(h, [-1])


class ValueNetwork(nn.Module):
    """Value Network.

    :param gym.spaces.Box or int observation_space: The observation space information. It is an instance
        of class: ``gym.spaces.Box``. It can also be an integer which represents the dimension of the observation.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            observation_space: Union[Box, Space, int],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
            norm_layer: Optional[ModuleType] = None,
    ) -> None:
        super(ValueNetwork, self).__init__()
        self._device = device
        if isinstance(observation_space, int):
            state_dim = observation_space
        else:
            state_dim = observation_space.shape[0]
        output_dim = 1
        self._layers = []
        hidden_sizes = [state_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, norm_layer, nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], output_dim)]
        self._model = nn.Sequential(*self._layers)

    def forward(self, inputs: Union[np.ndarray, Tensor]) -> Tensor:
        inputs = torch.as_tensor(inputs, device=self._device, dtype=torch.float32)
        h = self._model(inputs)
        return torch.reshape(h, [-1])


# class SinusoidalPosEmb(nn.Module):
#     """Sinusoidal positional embedding for time steps."""
#
#     def __init__(self, dim: int):
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         device = x.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb
#
#
# class MLPResNetBlock(nn.Module):
#     """
#     MLP ResNet Block with Layer Normalization and Dropout.
#     Architecture based on IDQL Paper Appendix G[cite: 533, 534].
#     Structure: Input -> Dropout -> LayerNorm -> Dense(4*h) -> Act -> Dense(h) -> Add
#     """
#
#     def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.ln = nn.LayerNorm(hidden_dim)
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
#         self.activation = nn.Mish()  # Mish usually performs better in diffusion, or use ReLU
#         self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         residual = x
#         out = self.dropout(x)
#         out = self.ln(out)
#         out = self.fc1(out)
#         out = self.activation(out)
#         out = self.fc2(out)
#         return residual + out
#
#
# class DiffusionActor(nn.Module):
#     """
#     Diffusion-based Actor Network for IDQL.
#     Models the behavior policy mu(a|s) using a DDPM.
#
#     :param observation_space: gym.spaces.Box
#     :param action_space: gym.spaces.Box
#     :param hidden_dim: Hidden dimension size for the MLP
#     :param num_res_blocks: Number of residual blocks (IDQL uses 3) [cite: 533]
#     :param time_steps: Number of diffusion steps T (IDQL uses small T, e.g., 5) [cite: 501]
#     :param dropout_rate: Dropout rate (IDQL uses 0.1) [cite: 480]
#     """
#
#     def __init__(
#             self,
#             observation_space: Union[Box, Space],
#             action_space: Union[Box, Space],
#             hidden_dim: int = 256,
#             num_res_blocks: int = 3,
#             time_steps: int = 5,
#             dropout_rate: float = 0.1,
#             device: Union[str, int, torch.device] = 'cpu',
#     ):
#         super().__init__()
#         self._device = device
#         self._action_space = action_space
#         self.state_dim = observation_space.shape[0]
#         self.action_dim = action_space.shape[0]
#         self.T = time_steps
#
#         # --- 1. Define Diffusion Noise Schedule (Variance Preserving / Beta Schedule) ---
#         # Using a linear schedule for simplicity, IDQL often uses VP schedule.
#         self.register_buffer('betas', torch.linspace(1e-4, 0.02, self.T).to(device))
#         self.register_buffer('alphas', 1. - self.betas)
#         self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
#
#         # --- 2. Define Network Architecture (LN_Resnet) ---
#         # Input projection: State + Action + Time Embedding -> Hidden
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Mish(),
#         )
#
#         self.input_proj = nn.Linear(self.state_dim + self.action_dim + hidden_dim, hidden_dim)
#
#         self.res_blocks = nn.ModuleList([
#             MLPResNetBlock(hidden_dim, dropout_rate) for _ in range(num_res_blocks)
#         ])
#
#         self.output_proj = nn.Linear(hidden_dim, self.action_dim)
#
#         self._action_means, self._action_mags = get_spec_means_mags(
#             self._action_space, self._device
#         )
#
#         self.to(device)
#
#     def forward(self, state: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         """
#         Predicts noise epsilon given state, noisy action, and time step.
#         """
#         # Embed time
#         t_emb = self.time_mlp(t)  # [Batch, Hidden]
#
#         # Concatenate inputs
#         x = torch.cat([state, action, t_emb], dim=-1)
#         x = self.input_proj(x)
#
#         # Pass through residual blocks
#         for block in self.res_blocks:
#             x = block(x)
#
#         return self.output_proj(x)
#
#     def get_diffusion_loss(self, state: Union[np.ndarray, torch.Tensor],
#                            action: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
#         """
#         Computes the MSE loss between added noise and predicted noise.
#         Used for training the behavior policy.
#         """
#         state = torch.as_tensor(state, dtype=torch.float32, device=self._device)
#         action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
#
#         # Normalize action to [-1, 1] for diffusion process
#         # action = (action - self._action_means) / self._action_mags
#
#         batch_size = state.shape[0]
#
#         # 1. Sample random time steps t ~ Uniform(0, T-1)
#         t = torch.randint(0, self.T, (batch_size,), device=self._device).long()
#
#         # 2. Sample random noise epsilon
#         noise = torch.randn_like(action, device=self._device)
#
#         # 3. Compute noisy action x_t (Forward Diffusion)
#         # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
#         alpha_bar_t = self.alpha_bars[t].view(-1, 1)
#         noisy_action = torch.sqrt(alpha_bar_t) * action + torch.sqrt(1 - alpha_bar_t) * noise
#
#         # 4. Predict noise using the network
#         predicted_noise = self(state, noisy_action, t)
#
#         # 5. Compute MSE Loss
#         loss = F.mse_loss(predicted_noise, noise)
#         return loss
#
#     @torch.no_grad()
#     def sample_n(self, state: Union[np.ndarray, torch.Tensor], n: int = 1) -> torch.Tensor:
#         """
#         Samples N actions per state using the reverse diffusion process.
#         Used during IDQL inference to generate candidates for Argmax.
#         """
#         state = torch.as_tensor(state, dtype=torch.float32, device=self._device)
#         batch_size = state.shape[0]
#
#         # Repeat state N times: [B, D] -> [B*N, D]
#         # We process (Batch * N) samples in parallel
#         state_rep = torch.repeat_interleave(state, n, dim=0)
#
#         # Start from pure Gaussian noise: x_T ~ N(0, I)
#         x = torch.randn((batch_size * n, self.action_dim), device=self._device)
#
#         # Reverse Diffusion Process: T-1 -> 0
#         for i in reversed(range(self.T)):
#             t = torch.full((batch_size * n,), i, device=self._device, dtype=torch.long)
#
#             # Predict noise
#             predicted_noise = self(state_rep, x, t)
#
#             # Compute alpha, beta, alpha_bar for step t
#             alpha = self.alphas[i]
#             alpha_bar = self.alpha_bars[i]
#             beta = self.betas[i]
#
#             # Additional noise z (only if t > 0)
#             if i > 0:
#                 z = torch.randn_like(x)
#             else:
#                 z = torch.zeros_like(x)
#
#             # Update x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_bar) * eps) + sigma * z
#             noise_factor = (1 - alpha) / torch.sqrt(1 - alpha_bar)
#             mean = (1 / torch.sqrt(alpha)) * (x - noise_factor * predicted_noise)
#             sigma = torch.sqrt(beta)  # Simple sigma choice
#
#             x = mean + sigma * z
#
#         # Un-normalize actions back to environment space
#         # x is currently in [-1, 1] approx (depending on scheduler)
#         # Clip to ensure valid range before scaling
#         x = x.clamp(-1, 1)
#         # x = x * self._action_mags + self._action_means
#
#         # Reshape back to [Batch, N, Action_Dim] for easier downstream processing
#         return x.view(batch_size, n, self.action_dim)
#
#     @property
#     def action_space(self) -> Box:
#         return self._action_space

class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, learnable: bool = True):
        super().__init__()
        assert output_dim % 2 == 0
        self.learnable = learnable
        self.output_dim = output_dim

        if learnable:
            self.W = nn.Parameter(
                torch.randn(output_dim // 2, input_dim) * 0.2
            )
        else:
            self.register_buffer("W", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)
        """
        if self.learnable:
            f = 2 * math.pi * x @ self.W.t()
        else:
            half_dim = self.output_dim // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(half_dim, device=x.device) / (half_dim - 1)
            )
            f = x * freqs

        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class MLP1(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, act=nn.SiLU):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), act()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPResNetBlock(nn.Module):
    def __init__(self, features, act=nn.SiLU, dropout_rate=None, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(features)

        # JAX 代码中是 Dense(features * 4) -> act -> Dense(features)
        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.act = act()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None

    def forward(self, x, training=True):
        residual = x
        if self.dropout and training:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.ln(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        # 如果维度不匹配，JAX 版会做一次线性映射（这里 features 保持一致，通常直接加）
        return residual + x


class MLPResNet(nn.Module):
    def __init__(self, input_dim, num_blocks, out_dim, hidden_dim=256, act=nn.SiLU, dropout_rate=None,
                 use_layer_norm=False):
        super().__init__()
        self.initial_dense = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MLPResNetBlock(hidden_dim, act, dropout_rate, use_layer_norm)
            for _ in range(num_blocks)
        ])
        self.final_act = act()
        self.final_dense = nn.Linear(hidden_dim, out_dim)

        # 对齐 JAX 的 xavier_uniform 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, training=True):
        x = self.initial_dense(x)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.final_act(x)
        return self.final_dense(x)


class DiffusionActorNetwork(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            T: int = 5,
            time_embed_dim: int = 128,
            cond_hidden_dims=(256, 256),
            denoise_hidden_dims=(256, 256),
            beta_schedule: str = "vp",
            clip_action: bool = True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.T = T
        self.clip_action = clip_action

        # ---- time embedding ----
        self.time_embed = FourierFeatures(1, time_embed_dim, learnable=True)

        # ---- condition encoder ----
        self.cond_encoder = MLP1(
            time_embed_dim,
            cond_hidden_dims,
            time_embed_dim,
        )

        # ---- denoise network ----
        denoise_input_dim = act_dim + obs_dim + time_embed_dim
        self.denoise_net = MLPResNet(
            input_dim=act_dim + obs_dim + time_embed_dim,
            num_blocks=3,  # 对应 actor_num_blocks
            out_dim=act_dim,
            hidden_dim=256,
            use_layer_norm=True,  # 对应 actor_layer_norm
            dropout_rate=0.1  # 对应 actor_dropout_rate
        )

        # ---- diffusion schedule ----
        self.register_buffer("betas", self._make_beta_schedule(beta_schedule))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer(
            "alpha_hats",
            torch.cumprod(self.alphas, dim=0),
        )

    def _make_beta_schedule(self, schedule: str) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(1e-4, 2e-2, self.T)
        elif schedule == "vp":
            t = torch.arange(1, self.T + 1)
            T = self.T
            b_min, b_max = 0.1, 10.0
            alpha = torch.exp(
                -b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / (T ** 2)
            )
            return 1 - alpha
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def forward(
            self,
            obs: torch.Tensor,
            noisy_act: torch.Tensor,
            t: torch.Tensor,
    ) -> torch.Tensor:
        """
        obs: (B, obs_dim)
        noisy_act: (B, act_dim)
        t: (B, 1)
        """
        t_embed = self.time_embed(t.float())
        cond = self.cond_encoder(t_embed)

        x = torch.cat([noisy_act, obs, cond], dim=-1)
        eps_pred = self.denoise_net(x)
        return eps_pred

    @torch.no_grad()
    def sample(
            self,
            obs: torch.Tensor,
            temperature: float = 0.01,
            repeat_last_step: int = 0,
    ) -> torch.Tensor:
        B = obs.size(0)
        device = obs.device
        x = torch.randn(B, self.act_dim, device=device)

        # 主循环
        for t in reversed(range(self.T)):
            x = self._denoise_step(x, obs, t, temperature)
            if self.clip_action:
                x = torch.clamp(x, -1.0, 1.0)

        # 👈 对应 JAX 的 for _ in range(repeat_last_step): fn(tuple, 0)
        for _ in range(repeat_last_step):
            # 重复步不加噪声 (temperature 设为 0 或利用内部 time > 0 判断)
            x = self._denoise_step(x, obs, 0, temperature=0.0)
            if self.clip_action:
                x = torch.clamp(x, -1.0, 1.0) # JAX 的 fn 内部每步都有 clip

        return x

    def _denoise_step(self, x, obs, t, temperature):
        B = x.size(0)
        t_tensor = torch.full((B, 1), t, device=x.device, dtype=torch.float32)
        eps = self.forward(obs, x, t_tensor)

        alpha = self.alphas[t]
        alpha_hat = self.alpha_hats[t]
        beta = self.betas[t]

        # 确定性部分
        x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * eps
        )

        # 随机部分 (仅在 t > 0 时)
        if t > 0 and temperature > 0:
            noise = torch.randn_like(x) * temperature
            x += torch.sqrt(beta) * noise
        return x
