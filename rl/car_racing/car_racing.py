import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os 

#
model_path = "car_racing_model.zip"
checkpoint_dir = "./checkpoints/"
TOTAL_TIMESTEPS = 50_000_000
N_ENVS = 8
SAVE_FREQ = 1_000_000

os.makedirs(checkpoint_dir, exist_ok=True)

if os.path.exists(model_path):
    model = PPO.load(model_path)
    print("Model loaded from", model_path)

    env = make_vec_env("CarRacing-v3", n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=4)
    model.set_env(env)

else:
    train_env = make_vec_env("CarRacing-v3", n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    train_env = VecFrameStack(train_env, n_stack=4)

    eval_env = make_vec_env("CarRacing-v3", n_envs=1, vec_env_cls=SubprocVecEnv)
    eval_env = VecFrameStack(eval_env, n_stack=4)


    callback = CheckpointCallback(
        save_freq=SAVE_FREQ // N_ENVS,
        save_path=checkpoint_dir,
        name_prefix="ppo_carracing"
    )

    model = PPO(
        "CnnPolicy", 
        train_env, 
        verbose=1,
        device='cuda',
        batch_size=256,
        ent_coef=0.01,
        learning_rate=3e-4
        )
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        model.save(model_path)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save(model_path)
    finally:
        train_env.close()
        eval_env.close()


