import gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env('ALE/Pong-v5', n_envs=1, seed=42)  
env = VecFrameStack(env, n_stack=4)

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_pong_tensorboard/")
model.learn(total_timesteps=1_000_000)

model.save("ppo_pong_agent")
env.close()

# Github.com/RezaGooner
