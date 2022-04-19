import logging
import os

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from bandit.env import DebuggingEnv
from stable_baselines3.common.evaluation import evaluate_policy

def main(args):
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    # Obtain the arguments/parameters for the policy and create the PPO model
    model = PPO.load(args.path)
    eval_env = DebuggingEnv(debug=True)

    successes = 0
    total = 1000
    # Train the model for the given number of steps
    for _ in range(total):
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=20)
        if mean_reward == 1:
            successes += 1
    print(f"Success rate: {successes/total}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        required=True,
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
