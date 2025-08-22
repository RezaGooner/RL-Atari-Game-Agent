import time
import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import AtariPreprocessing, TransformObservation

gym.register_envs(ale_py)

def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=1)
    
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False
    )

    new_obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(1, 84, 84),
        dtype=np.uint8
    )
    env = TransformObservation(
        env,
        lambda obs: np.expand_dims(obs, axis=2).transpose(2, 0, 1),
        new_obs_space
    )
    return env

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4, channels_order="first")

print("Run observation space:", env.observation_space)
model = DQN.load("breakout_dqn_84x84.zip", env=env)

obs = env.reset()
for _ in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    time.sleep(0.01)

env.close()

# Github.com/RezaGooner
