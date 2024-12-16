import os
import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from huggingface_sb3 import package_to_hub
from huggingface_hub import login, whoami
from huggingface_hub.errors import LocalTokenNotFoundError

from rl_course.log import logger


def load_dotenv(env_path: str) -> None:
    loaded_vars = []
    try:
        with open(env_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty lines and comments
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    if "=" not in line:
                        logger.info(f"Warning: Invalid format at line {line_num}: {line}")
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if value and value[0] == value[-1] and value[0] in "\"'":
                        value = value[1:-1]
                    if not key.isidentifier():
                        logger.info(
                            f"Warning: Invalid variable name at line {line_num}: {key}"
                        )
                        continue

                    os.environ[key] = value
                    loaded_vars.append(key)

                except Exception as e:
                    logger.info(f"Warning: Could not process line {line_num}: {e}")

    except FileNotFoundError:
        logger.info(f"Error: Environment file '{env_path}' not found")
        return {}
    except Exception as e:
        logger.info(f"Error: Failed to load environment file: {e}")
        return {}
    logger.info(f"Loaded environment variables: {loaded_vars}")


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def show_model(env, model) -> None:
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            break
    env.close()


def create_env(env_name: str, seed: int, render_mode: str = "rgb_array") -> gym.Env:
    """Create and seed environment."""
    env = gym.make(env_name, render_mode=render_mode)
    env = Monitor(env)
    env.reset(seed=seed)  # New gymnasium API
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def evaluate_model(
    env_name: str, model, num_episodes: int = 10, seed: int = 5136
) -> Dict:
    set_seeds(seed)
    eval_env = create_env(env_name, seed)
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=num_episodes, deterministic=True
    )
    eval_env.close()
    return {"mean_reward": mean_r, "std_reward": std_r}


def submit_to_hg_hub(
    model: Any,
    model_name: str,
    env_id: str,
    model_arch: str,
    repo_id: str,
    commit_message: str,
) -> None:
    """Submit model to Hugging Face Hub
    Args:
        model: The trained model
        model_name: Name of the model
        env_id: Environment ID
        model_arch: Model architecture
        repo_id: id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name}
        commit_message: Commit message
    """
    try:
        _ = whoami()
    except LocalTokenNotFoundError:
        hg_token = os.getenv("HG_TOKEN", None)
        if hg_token is None:
            raise ValueError(
                "HG_TOKEN is not set. Please set it in the environment variables."
            )
        login(token=hg_token)

    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])
    package_to_hub(
        model=model,
        model_name=model_name,
        model_architecture=model_arch,
        env_id=env_id,
        eval_env=eval_env,
        repo_id=repo_id,
        commit_message=commit_message,
    )
