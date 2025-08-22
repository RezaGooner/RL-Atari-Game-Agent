import gymnasium as gym
import pygame
import ale_py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

env_for_model = make_atari_env("ALE/Pong-v5", n_envs=1)
env_for_model = VecFrameStack(env_for_model, n_stack=4)

env_render = gym.make("ALE/Pong-v5", render_mode="human")

model = PPO.load("ppo_pong_agent", device="cpu")

pygame.init()
pygame.display.set_caption("Pong: Human vs Agent")
clock = pygame.time.Clock()
running = True

obs_render, _ = env_render.reset()
obs_model = env_for_model.reset()

action_human = 0

while running:
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        action_human = 2
    elif keys[pygame.K_s]:
        action_human = 5
    else:
        action_human = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action_agent, _ = model.predict(obs_model, deterministic=True)
    action_agent = action_agent.item()

    obs_render, reward_render, terminated_r, truncated_r, info_r = env_render.step(action_human)
    obs_model, reward_model, done_m, info_m = env_for_model.step([action_agent])

    if terminated_r or truncated_r:
        obs_render, _ = env_render.reset()
        obs_model = env_for_model.reset()

    clock.tick(60)
env_render.close()
pygame.quit()

# Github.com/RezaGooner
