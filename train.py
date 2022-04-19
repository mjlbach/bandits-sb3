import os
import logging

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from bandit.env import make_env

from bandit.model import CustomCombinedExtractor


def main(args):
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    num_environments = 8

    # Multiprocess
    env = SubprocVecEnv([make_env(i) for i in range(num_environments)])
    env = VecMonitor(env)

    # Obtain the arguments/parameters for the policy and create the PPO model
    save_dir = "experiments"
    experiment_name = args.name
    log_dir = os.path.join(save_dir, experiment_name, "log")
    checkpoints_dir = os.path.join(save_dir, experiment_name, "checkpoints")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=1e-5, # This is key
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
    )
    print(model.policy)

    # Train the model for the given number of steps
    total_timesteps = 1000
    for i in range(100):
        model.learn(total_timesteps, reset_num_timesteps=False)
        # Save the trained model and delete it
        model.save(os.path.join(checkpoints_dir, f"ckpt-{i*total_timesteps}"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        "-n",
        default="test",
        help="name to save experiment under",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
