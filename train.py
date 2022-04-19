import logging
import os
from typing import Callable

import torch as th
import torch.nn as nn
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from bandit.env import DebuggingEnv

from gym.spaces import Dict


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            if key in ["vectorized_goal", "task_obs"]:
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU())
            elif key in ["rgb"]:
                n_input_channels = subspace.shape[2]  # channel last
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key in ["scan"]:
                n_input_channels = subspace.shape[1]  # channel last
                cnn = nn.Sequential(
                    nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[1], subspace.shape[0]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            else:
                raise ValueError("Unknown observation key: %s" % key)
            total_concat_size += feature_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                observations[key] = observations[key].permute((0, 3, 1, 2)) #type: ignore
            elif key in ["scan"]:
                observations[key] = observations[key].permute((0, 2, 1)) #type: ignore
            encoded_tensor_list.append(extractor(observations[key])) #type: ignore
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


def main():
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    tensorboard_log_dir = "log_dir"
    num_environments = 8

    # Function callback to create environments
    def make_env(rank: int, seed: int = 0) -> Callable:
        def _init():
            env = DebuggingEnv()
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    # Multiprocess
    env = SubprocVecEnv([make_env(i) for i in range(num_environments)])
    env = VecMonitor(env)

    # Obtain the arguments/parameters for the policy and create the PPO model
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs)
    print(model.policy)

    # Train the model for the given number of steps
    total_timesteps = 1000
    for i in range(100):
        model.learn(total_timesteps)
        # Save the trained model and delete it
        model.save(f"ckpt-{i*total_timesteps}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
