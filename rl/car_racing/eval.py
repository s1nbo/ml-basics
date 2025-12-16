import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Evaluation
model_path = "car_racing_model.zip"

model = PPO.load(model_path)
print("Model loaded from", model_path)
env = make_vec_env("CarRacing-v3", n_envs=1, vec_env_cls=SubprocVecEnv)
env = VecFrameStack(env, n_stack=4)
obs = env.reset()

for i in range(3):
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print("Total Reward:", total_reward)