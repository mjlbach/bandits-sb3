import logging
import os

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from bandit.env import DebuggingEnv
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    # Obtain the arguments/parameters for the policy and create the PPO model
    model = PPO.load("save/ckpt-9000")
    eval_env = DebuggingEnv(debug=True)

    # Train the model for the given number of steps
    for i in range(100):
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
        print(mean_reward)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
