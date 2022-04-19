import re
from pathlib import Path

from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
import os

# from ssg.utils.callbacks import MetricsCallback

from bandit.flat_model import ComplexInputNetworkFlat
from bandit.debug.env import env_creator

ModelCatalog.register_custom_model("graph_extractor", ComplexInputNetworkFlat)

def main():
    # instantiate env class
    register_env("env_creator", env_creator)

    n_steps = 4096
    num_envs = 8

    experiment_save_path = os.path.expanduser("~/ray_results")
    experiment_name = "test_mab_fcnet"

    checkpoint_path = Path(experiment_save_path, experiment_name)

    config = {
        "env": "env_creator",
        "model": {
          "custom_model": "graph_extractor",
          "post_fcnet_hiddens": [256, 256, 256],
          "conv_filters": [[16, [4, 4], 4], [32, [4, 4], 4], [256, [8, 8], 2]]
        },
        "num_workers": 0,
        "framework": "torch",
        "seed": 0,
        "lambda": 0.9,
        "lr": 1e-4,
        "train_batch_size": n_steps,
        "rollout_fragment_length":  n_steps // num_envs,
        "num_sgd_iter": 30,
        "sgd_minibatch_size": 128,
        "gamma": 0.5,
        "create_env_on_driver": False,
        "num_gpus": 0,
        # "callbacks": MetricsCallback,
        # "log_level": "DEBUG",
        # "_disable_preprocessor_api": False,
    }

    log_path = str(checkpoint_path.joinpath("eval"))
    Path(log_path).mkdir(parents=True, exist_ok=True)
    agent = ppo.PPOTrainer(
        config,
        logger_creator=lambda x: UnifiedLogger(x, log_path), #type: ignore
    )

    if Path(checkpoint_path).exists():
        checkpoints = Path(checkpoint_path).rglob("checkpoint-*")
        checkpoints = [
            str(f) for f in checkpoints if re.search(r".*checkpoint-\d*$", str(f))
        ]
        checkpoints = sorted(checkpoints)
        if len(checkpoints) > 0:
            agent.restore(checkpoints[-1])

    env = env_creator('test')
    state = env.reset()
    trials = 0
    successes = 0

    for _ in range(10):
        episode_reward = 0
        done = False
        success = False

        state = env.reset()
        while not done:
            action = agent.compute_single_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        trials += 1
        print()
        print("Episode finished:")
        print("-" * 30)
        print(f"{'Success:': <20} {success: b}")
        print(f"{'Episode reward': <20} {episode_reward: .2f}")
        print("-" * 30)

    print()
    print(f"Success fraction: {successes/trials}")


if __name__ == "__main__":
    main()
