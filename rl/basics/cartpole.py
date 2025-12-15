import gymnasium as gym
from stable_baselines3 import PPO
import os 

# Training
model_path = "cartpole_model.zip"

if os.path.exists(model_path):
    model = PPO.load(model_path)
    print("Model loaded from", model_path)
else:
    train_env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", train_env, verbose=0)
    model.learn(total_timesteps=50000)
    train_env.close()
    model.save(model_path)
    print("Training complete!")



# Evaluation
env = gym.make("CartPole-v1", render_mode="human")

for i in range(3):
    obs, info = env.reset()

    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action, )
        total_reward += reward
        done = terminated or truncated


    print("Total Reward:", total_reward)