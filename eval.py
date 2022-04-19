import logging
import os

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from bandit.env import make_env
from bandit.model import CustomCombinedExtractor

def main():
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    tensorboard_log_dir = "log_dir"
    num_environments = 8

    # Function callback to create environments
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
