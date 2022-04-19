import logging

from stable_baselines3.ppo.ppo import PPO
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
    eval_env = DebuggingEnv(debug=False)

    # Train the model for the given number of steps
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)
    print(f"Mean reward: {mean_reward}")
    print(f"Std reward: {std_reward}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        required=True,
        help="path of ckpt zip to load",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
