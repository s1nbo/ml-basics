import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
import os 

#
model_path = "car_racing_model.zip"
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 8

if os.path.exists(model_path):
    model = PPO.load(model_path)
    print("Model loaded from", model_path)
else:
    train_env = make_vec_env("CarRacing-v3", n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    train_env = VecFrameStack(train_env, n_stack=4)
    model = PPO(
        "CnnPolicy", 
        train_env, 
        verbose=1,
        device='cuda',
        batch_size=256,
        ent_coef=0.01,
        learning_rate=3e-4
        )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_path)
    train_env.close()
   
    print("Training complete!")

# Evaluation
env = gym.make("CarRacing-v3", n_envs=1, env_kwargs={"render_mode": "human"})
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