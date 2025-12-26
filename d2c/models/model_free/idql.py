import collections
import copy

import numpy as np
import torch
import torch.nn.functional as F
from gym import Space
from gym.spaces import Box
from torch import nn, Tensor
from typing import Dict, Tuple, Any, Union, Sequence, List, Optional, Callable

from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import networks, utils, policies

def default_init(scale: Optional[float] = None):
    """对齐 JAX 的 xavier_uniform 初始化"""
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if scale is not None:
                m.weight.data *= scale
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return init_weights

class ValueNetwork(nn.Module):
    def __init__(
            self,
            observation_space: Union[int, any],
            fc_layer_params: Sequence[int],
            device: Union[str, torch.device] = "cpu",
            use_layer_norm: bool = False,
            dropout_rate: Optional[float] = None,
            activations: Callable = F.relu,
    ):
        super().__init__()
        self._device = device
        self._activations = activations

        if isinstance(observation_space, int):
            state_dim = observation_space
        else:
            state_dim = observation_space.shape[0]

        # ---- 1. 输入层 LayerNorm ----
        self.input_ln = nn.LayerNorm(state_dim) if use_layer_norm else nn.Identity()

        # ---- 2. MLP 主体 (Base Class) ----
        # 对应 JAX 的 MLP(hidden_dims=critic_hidden_dims, activate_final=True)
        mlp_layers = []
        dims = [state_dim] + list(fc_layer_params)

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i+1]
            layer = nn.Linear(in_d, out_d)
            mlp_layers.append(layer)

            # 最后一层是否激活？由于 activate_final=True，这里所有层都激活
            # 对应 JAX: if i + 1 < len(self.hidden_dims) or self.activate_final:
            if dropout_rate is not None and dropout_rate > 0:
                mlp_layers.append(nn.Dropout(p=dropout_rate))

            # 注意：PyTorch 的 nn.ReLU 不能传 Callable，这里我们用自定义包装或者直接用组件
            mlp_layers.append(self._get_activation_module(activations))

        self.base_mlp = nn.Sequential(*mlp_layers)

        # ---- 3. StateValue 输出层 ----
        # JAX 的 StateValue 通常是在 base_cls 之后再接一个 Dense(1)
        self.value_head = nn.Linear(fc_layer_params[-1], 1)

        # ---- 4. 初始化 ----
        self.apply(default_init())
        self.to(device)

    def _get_activation_module(self, act_fn):
        if act_fn == F.relu: return nn.ReLU()
        if act_fn == F.silu or act_fn == torch.nn.functional.silu: return nn.SiLU()
        # 默认返回 ReLU
        return nn.ReLU()

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        # 确保输入在正确的设备上
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self._device, dtype=torch.float32)

        # 切换训练/评估模式以控制 Dropout
        if training:
            self.train()
        else:
            self.eval()

        x = self.input_ln(x)
        x = self.base_mlp(x)
        v = self.value_head(x)
        return v.squeeze(-1)


class IDQLAgent(BaseAgent):
    """
    Implicit Diffusion Q-Learning (IDQL)
    """

    def __init__(
            self,
            expectile: float = 0.5,
            temperature: float = 1.0,
            ddpm_temperature: float = 0,
            n_action_samples: int = 64,
            **kwargs: Any,
    ):
        self._expectile = expectile
        # self._temperature = temperature
        self._ddpm_temperature = ddpm_temperature
        self._n_action_samples = n_action_samples
        super().__init__(**kwargs)

    def _build_fns(self) -> None:
        self._agent_module = IDQLAgentModule(modules=self._modules)

        self._v_fn = self._agent_module.v_net
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets

        self._actor_score = self._agent_module.actor
        self._actor_score_target = self._agent_module.actor_target

    def _build_optimizers(self) -> None:
        opts = self._optimizers

        self._v_optimizer = utils.get_optimizer(opts.v[0])(
            self._v_fn.parameters(),
            lr=opts.v[1],
            weight_decay=self._weight_decays,
        )

        self._q_optimizer = utils.get_optimizer(opts.q[0])(
            self._q_fns.parameters(),
            lr=opts.q[1],
            weight_decay=self._weight_decays,
        )

        self._actor_optimizer = utils.get_optimizer('adamw')(
            self._actor_score.parameters(),
            lr=opts.p[1],
            weight_decay=self._weight_decays,
        )

        decay_steps = 3000000
        self._actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._actor_optimizer,
            T_max=decay_steps,
            eta_min=0.0 # 学习率最终降至 0
        )

    def _build_v_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s = batch['s1']
        a = batch['a1']

        with torch.no_grad():
            q1 = self._q_target_fns[0](s, a)
            q2 = self._q_target_fns[1](s, a)
            q = torch.min(q1, q2)

        v = self._v_fn(s)
        diff = q - v
        weight = torch.where(diff > 0, self._expectile, 1 - self._expectile)
        v_loss = (weight * diff.pow(2)).mean()

        return v_loss, {
            "v_loss": v_loss.detach(),
            "V": v.mean().detach(),
        }

    def _build_q_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s1, s2 = batch['s1'], batch['s2']
        a = batch['a1']
        r = batch['reward']
        dsc = batch['dsc']

        with torch.no_grad():
            next_v = self._v_fn(s2)
            target_q = r + dsc * self._discount * next_v

        q1 = self._q_fns[0](s1, a)
        q2 = self._q_fns[1](s1, a)

        q_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))

        return q_loss, {
            "q_loss": q_loss.detach(),
            "Q": q1.mean().detach(),
        }

    def _build_p_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        self._actor_score.train()
        s = batch['s1']
        a = batch['a1']
        # print("s: ", s)
        # print("a: ", a)
        # sample diffusion timestep
        t = torch.randint(
            0, self._actor_score.T, (a.size(0), 1), device=self._device
        )

        noise = torch.randn_like(a)

        alpha_hat = self._actor_score.alpha_hats[t].to(self._device)
        noisy_a = torch.sqrt(alpha_hat) * a + torch.sqrt(1 - alpha_hat) * noise

        eps_pred = self._actor_score(s, noisy_a, t)

        actor_loss = F.mse_loss(eps_pred, noise)

        return actor_loss, {
            "actor_loss": actor_loss.detach(),
        }



    def _optimize_step(self, batch: Dict) -> Dict:
        info = collections.OrderedDict()

        v_loss, v_info = self._build_v_loss(batch)
        self._v_optimizer.zero_grad()
        v_loss.backward()
        self._v_optimizer.step()

        q_loss, q_info = self._build_q_loss(batch)
        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        p_loss, p_info = self._build_p_loss(batch)
        self._actor_optimizer.zero_grad()
        p_loss.backward()
        self._actor_optimizer.step()

        self._actor_scheduler.step()

        # 记录当前学习率 (可选，用于 Debug)
        current_lr = self._actor_optimizer.param_groups[0]['lr']
        p_info["actor_lr"] = torch.tensor(current_lr)

        # Target 网络更新
        self._update_target_fns(self._q_fns, self._q_target_fns)
        # utils.soft_update(self._actor_score, self._actor_score_target, tau=0.005)

        info.update(v_info)
        info.update(q_info)
        info.update(p_info)
        return info

    # def _sample_actions(self, s: Tensor) -> Tensor:
    #     """
    #     Aligned with JAX ddpm_sampler and eval_actions logic.
    #     s: (obs_dim,) or (B, obs_dim)
    #     """
    #     # 1. 确保输入是 Batch 形式 (B, obs_dim)
    #     if s.dim() == 1:
    #         s = s.unsqueeze(0)
    #
    #     batch_size = s.size(0)
    #     device = s.device
    #
    #     # 2. 切换到 eval 模式以关闭 Dropout
    #     self._actor_score_target.eval()
    #     self._q_target_fns.eval()
    #
    #     with torch.no_grad():
    #         # 3. 对每个 observation 重复采样 N 个动作
    #         # 结果形状: (B * N, obs_dim)
    #         s_rep = s.repeat_interleave(self._n_action_samples, dim=0)
    #
    #         # 4. 调用扩散模型采样
    #         # 传入 repeat_last_step (对应 JAX 里的 M)
    #         actions = self._actor_score_target.sample(
    #             s_rep,
    #             temperature=self._ddpm_temperature,
    #             repeat_last_step=self._config.rl_config.get('M', 0)
    #         )
    #
    #         # 5. 计算这些动作的 Q 值
    #         # q_target_fns 是 Ensemble，通常取两个 Q 的最小值 (Clipping)
    #         q1 = self._q_target_fns[0](s_rep, actions)
    #         q2 = self._q_target_fns[1](s_rep, actions)
    #         q_values = torch.min(q1, q2) # (B * N,)
    #
    #         # 6. 从每个 state 对应的 N 个采样中选出最优动作
    #         # 先重塑为 (B, N)
    #         q_values = q_values.view(batch_size, self._n_action_samples)
    #         actions = actions.view(batch_size, self._n_action_samples, -1)
    #
    #         # 找到每个 Batch 中 Q 最大的索引
    #         best_idx = torch.argmax(q_values, dim=1) # (B,)
    #
    #         # 提取最优动作
    #         # 这里的 gathering 逻辑: actions[i, best_idx[i]]
    #         best_actions = actions[torch.arange(batch_size), best_idx]
    #
    #     # 如果输入是单条数据，返回 (act_dim,)；否则返回 (B, act_dim)
    #     return best_actions.squeeze(0) if batch_size == 1 else best_actions

    def _get_modules(self) -> utils.Flags:
        model_params_v = self._model_params.v[0]
        model_params_q, n_q_fns = self._model_params.q

        model_params_p = self._model_params.p[0]

        obs_dim = self._observation_space.shape[0]
        act_dim = self._action_space.shape[0]

        def v_net_factory():
            return ValueNetwork(
                observation_space=self._observation_space,
                fc_layer_params=model_params_v,
                device=self._device,
                use_layer_norm=True
            )

        def q_net_factory():
            return networks.CriticNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_q,
                device=self._device,
            )

        def actor_factory():
            """
            model_params_p 约定格式，例如：
            {
                "T": 5,
                "time_embed_dim": 128,
                "cond_hidden_dims": [256, 256],
                "denoise_hidden_dims": [256, 256],
                "beta_schedule": "vp",
                "clip_action": True,
            }
            """
            return networks.DiffusionActorNetwork(
                obs_dim=obs_dim,
                act_dim=act_dim,
                T=20,
                time_embed_dim=128,
                cond_hidden_dims=(256, 256),
                denoise_hidden_dims=(256, 256),
                beta_schedule="linear",
                clip_action=True,
            )

        modules = utils.Flags(
            v_net_factory=v_net_factory,
            q_net_factory=q_net_factory,
            actor_factory=actor_factory,
            n_q_fns=n_q_fns,
            device=self._device,
        )
        return modules

    def _build_test_policies(self) -> None:
        class IDQLTestPolicy:
            def __init__(self, agent: "IDQLAgent"):
                self._agent = agent

            def __call__(self, obs):
                if not torch.is_tensor(obs):
                    obs = torch.as_tensor(
                        obs, device=self._agent._device, dtype=torch.float32
                    )

                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)

                # sample N actions via diffusion
                obs_rep = obs.repeat(self._agent._n_action_samples, 1)
                actions = self._agent._actor_score_target.sample(
                    obs_rep,
                    temperature=self._agent._ddpm_temperature,
                )

                with torch.no_grad():
                    q1 = self._agent._q_target_fns[0](obs_rep, actions)
                    q2 = self._agent._q_target_fns[1](obs_rep, actions)
                    q = torch.min(q1, q2)

                # 将 Q 重塑为 (B, N)
                q = q.view(-1, self._agent._n_action_samples)
                # 找到每个样本最好的动作索引
                idx = q.argmax(dim=1) # (B,)
                # 重塑 actions 为 (B, N, act_dim)
                actions = actions.view(-1, self._agent._n_action_samples, self._agent._a_dim)
                # # 诊断 1：看看采样出来的动作是不是本身就在边界
                # print(f"Action stats: mean={actions.abs().mean():.3f}, max={actions.max():.3f}, min={actions.min():.3f}")
                #
                # # 诊断 2：看看 Q 值的分布
                # print(f"Q values: max={q.max():.3f}, min={q.min():.3f}, mean={q.mean():.3f}")
                # 提取
                best_action = actions[torch.arange(actions.size(0)), idx]
                if best_action.size(0) == 1:
                    # print(best_action)
                    return best_action.squeeze(0).cpu().numpy()
                else:
                    return best_action.cpu().numpy()

        self._test_policies["main"] = IDQLTestPolicy(self)

    def save(self, ckpt_name: str) -> None:
        torch.save(
            {
                "agent": self._agent_module.state_dict(),
                "actor": self._actor_score.state_dict(),
            },
            ckpt_name + ".pth",
            )

    def restore(self, ckpt_name: str) -> None:
        ckpt = torch.load(ckpt_name + ".pth", map_location=self._device)
        self._agent_module.load_state_dict(ckpt["agent"])
        self._actor_score.load_state_dict(ckpt["actor"])


class IDQLAgentModule(BaseAgentModule):

    def _build_modules(self) -> None:
        device = self._net_modules.device

        self._v_net = self._net_modules.v_net_factory().to(device)
        self._q_nets = nn.ModuleList([
            self._net_modules.q_net_factory().to(device)
            for _ in range(self._net_modules.n_q_fns)
        ])
        self._q_target_nets = copy.deepcopy(self._q_nets)

        self._actor = self._net_modules.actor_factory().to(device)
        self._actor_target = copy.deepcopy(self._actor)

    @property
    def v_net(self) -> nn.Module:
        return self._v_net

    @property
    def q_nets(self) -> nn.ModuleList:
        return self._q_nets

    @property
    def q_target_nets(self) -> nn.ModuleList:
        return self._q_target_nets

    @property
    def actor(self):
        return self._actor

    @property
    def actor_target(self):
        return self._actor_target
