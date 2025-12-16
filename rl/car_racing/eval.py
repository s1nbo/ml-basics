import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
import os


# Evaluation
if __name__ == "__main__":
    '''
    checkpoint_dir = "./checkpoints/"
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    model_path = os.path.join(checkpoint_dir, model_files[-1])
    '''
    model_path = 'car_racing_model.zip' 
    model = PPO.load(model_path)
    print("Model loaded from", model_path)
    env = make_vec_env(
        "CarRacing-v3", n_envs=1, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'render_mode': 'human'},
    )
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
    
    env.close()