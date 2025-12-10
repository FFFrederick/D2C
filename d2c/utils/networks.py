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
        hidden_sizes = [state_dim+self._action_dim] + list(fc_layer_params)
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
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], output_dim)]
        self._model = nn.Sequential(*self._layers)

    def forward(self, inputs: Union[np.ndarray, Tensor]) -> Tensor:
        inputs = torch.as_tensor(inputs, device=self._device, dtype=torch.float32)
        h = self._model(inputs)
        return torch.reshape(h, [-1])


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLPResNetBlock(nn.Module):
    """
    MLP ResNet Block with Layer Normalization and Dropout.
    Architecture based on IDQL Paper Appendix G[cite: 533, 534].
    Structure: Input -> Dropout -> LayerNorm -> Dense(4*h) -> Act -> Dense(h) -> Add
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.activation = nn.Mish()  # Mish usually performs better in diffusion, or use ReLU
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout(x)
        out = self.ln(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return residual + out


class DiffusionActor(nn.Module):
    """
    Diffusion-based Actor Network for IDQL.
    Models the behavior policy mu(a|s) using a DDPM.

    :param observation_space: gym.spaces.Box
    :param action_space: gym.spaces.Box
    :param hidden_dim: Hidden dimension size for the MLP
    :param num_res_blocks: Number of residual blocks (IDQL uses 3) [cite: 533]
    :param time_steps: Number of diffusion steps T (IDQL uses small T, e.g., 5) [cite: 501]
    :param dropout_rate: Dropout rate (IDQL uses 0.1) [cite: 480]
    """

    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            hidden_dim: int = 256,
            num_res_blocks: int = 3,
            time_steps: int = 5,
            dropout_rate: float = 0.1,
            device: Union[str, int, torch.device] = 'cpu',
    ):
        super().__init__()
        self._device = device
        self._action_space = action_space
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.T = time_steps

        # --- 1. Define Diffusion Noise Schedule (Variance Preserving / Beta Schedule) ---
        # Using a linear schedule for simplicity, IDQL often uses VP schedule.
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, self.T).to(device))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))

        # --- 2. Define Network Architecture (LN_Resnet) ---
        # Input projection: State + Action + Time Embedding -> Hidden
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        self.input_proj = nn.Linear(self.state_dim + self.action_dim + hidden_dim, hidden_dim)

        self.res_blocks = nn.ModuleList([
            MLPResNetBlock(hidden_dim, dropout_rate) for _ in range(num_res_blocks)
        ])

        self.output_proj = nn.Linear(hidden_dim, self.action_dim)

        self._action_means, self._action_mags = get_spec_means_mags(
            self._action_space, self._device
        )

        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predicts noise epsilon given state, noisy action, and time step.
        """
        # Embed time
        t_emb = self.time_mlp(t)  # [Batch, Hidden]

        # Concatenate inputs
        x = torch.cat([state, action, t_emb], dim=-1)
        x = self.input_proj(x)

        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x)

        return self.output_proj(x)

    def get_diffusion_loss(self, state: Union[np.ndarray, torch.Tensor],
                           action: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Computes the MSE loss between added noise and predicted noise.
        Used for training the behavior policy.
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=self._device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self._device)

        # Normalize action to [-1, 1] for diffusion process
        # action = (action - self._action_means) / self._action_mags

        batch_size = state.shape[0]

        # 1. Sample random time steps t ~ Uniform(0, T-1)
        t = torch.randint(0, self.T, (batch_size,), device=self._device).long()

        # 2. Sample random noise epsilon
        noise = torch.randn_like(action, device=self._device)

        # 3. Compute noisy action x_t (Forward Diffusion)
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)
        noisy_action = torch.sqrt(alpha_bar_t) * action + torch.sqrt(1 - alpha_bar_t) * noise

        # 4. Predict noise using the network
        predicted_noise = self(state, noisy_action, t)

        # 5. Compute MSE Loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample_n(self, state: Union[np.ndarray, torch.Tensor], n: int = 1) -> torch.Tensor:
        """
        Samples N actions per state using the reverse diffusion process.
        Used during IDQL inference to generate candidates for Argmax.
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=self._device)
        batch_size = state.shape[0]

        # Repeat state N times: [B, D] -> [B*N, D]
        # We process (Batch * N) samples in parallel
        state_rep = torch.repeat_interleave(state, n, dim=0)

        # Start from pure Gaussian noise: x_T ~ N(0, I)
        x = torch.randn((batch_size * n, self.action_dim), device=self._device)

        # Reverse Diffusion Process: T-1 -> 0
        for i in reversed(range(self.T)):
            t = torch.full((batch_size * n,), i, device=self._device, dtype=torch.long)

            # Predict noise
            predicted_noise = self(state_rep, x, t)

            # Compute alpha, beta, alpha_bar for step t
            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            beta = self.betas[i]

            # Additional noise z (only if t > 0)
            if i > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            # Update x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_bar) * eps) + sigma * z
            noise_factor = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            mean = (1 / torch.sqrt(alpha)) * (x - noise_factor * predicted_noise)
            sigma = torch.sqrt(beta)  # Simple sigma choice

            x = mean + sigma * z

        # Un-normalize actions back to environment space
        # x is currently in [-1, 1] approx (depending on scheduler)
        # Clip to ensure valid range before scaling
        x = x.clamp(-1, 1)
        # x = x * self._action_mags + self._action_means

        # Reshape back to [Batch, N, Action_Dim] for easier downstream processing
        return x.view(batch_size, n, self.action_dim)

    @property
    def action_space(self) -> Box:
        return self._action_space
