import sys
# sys.path.append('../../')
from pathlib import Path

# 获取当前脚本的绝对路径
current_file_path = Path(__file__).resolve()

# 假设项目根目录D2C在脚本目录(benchmark)的父目录的父目录
# current_file_path.parent 是 benchmark 目录
# current_file_path.parent.parent 是 example 目录
# current_file_path.parent.parent.parent 是 D2C 根目录
ROOT_DIR = current_file_path.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import torch
import logging
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.envs import benchmark_env, LeaEnv
from d2c.data import Data
from d2c.evaluators import bm_eval
from example.benchmark.config import make_config

logging.basicConfig(level=logging.INFO)


def main():
    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'd4rl',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': 'HalfCheetah-v2',
        prefix + 'data_name': 'halfcheetah_medium_expert-v2',
        prefix + 'state_normalize': True,
        prefix + 'score_normalize': True,
    }
    command_args.update({
        'model.model_name': 'iql',
        'train.data_loader_name': None,
        'train.device': device,
        'train.seed': 1,
        'train.total_train_steps': 1000000,
        'train.batch_size': 256,
        'train.agent_ckpt_name': '0810'
    })

    config = make_config(command_args)
    bm_data = Data(config)
    s_norm = dict(zip(['obs_shift', 'obs_scale'], bm_data.state_shift_scale))
    data = bm_data.data
    # The env of the benchmark to be used for policy evaluation.
    env = benchmark_env(config=config, **s_norm)
    # Contains dynamics model to be trained.
    lea_env = LeaEnv(config)
    agent = make_agent(config=config, env=lea_env, data=data)
    evaluator = bm_eval(agent=agent, env=env, config=config)
    trainer = Trainer(agent=agent, train_data=data, config=config, env=lea_env, evaluator=evaluator)
    trainer.train()


if __name__ == '__main__':
    main()
