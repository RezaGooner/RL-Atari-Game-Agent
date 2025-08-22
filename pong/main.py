import gymnasium as gym
import pygame
import ale_py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# ثبت محیط‌های ALE
gym.register_envs(ale_py)

# محیط مدل برای عامل
env_for_model = make_atari_env("ALE/Pong-v5", n_envs=1)
env_for_model = VecFrameStack(env_for_model, n_stack=4)

# محیط نمایش با کنترل انسانی
env_render = gym.make("ALE/Pong-v5", render_mode="human")

# بارگذاری مدل آموزش‌دیده
model = PPO.load("ppo_pong_agent", device="cpu")

# راه‌اندازی pygame
pygame.init()
pygame.display.set_caption("Pong: Human vs Agent")
clock = pygame.time.Clock()
running = True

# ریست اولیه
obs_render, _ = env_render.reset()
obs_model = env_for_model.reset()

# اکشن انسانی (پیش‌فرض توقف)
action_human = 0

# حلقه بازی Human-vs-Agent
while running:
    keys = pygame.key.get_pressed()

    # بررسی وضعیت کلیدها
    if keys[pygame.K_w]:
        action_human = 2  # حرکت به بالا
    elif keys[pygame.K_s]:
        action_human = 5  # حرکت به پایین
    else:
        action_human = 0  # توقف

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # پیش‌بینی عامل
    action_agent, _ = model.predict(obs_model, deterministic=True)
    action_agent = action_agent.item()

    # اجرای فقط یک قدم در هر فریم
    obs_render, reward_render, terminated_r, truncated_r, info_r = env_render.step(action_human)
    obs_model, reward_model, done_m, info_m = env_for_model.step([action_agent])

    # ریست اپیزود
    if terminated_r or truncated_r:
        obs_render, _ = env_render.reset()
        obs_model = env_for_model.reset()

    # افزایش نرخ فریم برای کنترل نرم‌تر
    clock.tick(60)  # تا 60 فریم در ثانیه

# پاکسازی
env_render.close()
pygame.quit()
