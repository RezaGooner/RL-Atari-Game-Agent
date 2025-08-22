import gymnasium as gym
import pygame
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import ale_py

# ثبت محیط‌های ALE
gym.register_envs(ale_py)

# ------------------------------
# بارگذاری مدل آموزش‌دیده
# ------------------------------
model = PPO.load("ppo_boxing_agent_1")

# ------------------------------
# ساخت محیط اصلی و محیط رندر جداگانه
# ------------------------------
def make_env():
    env = gym.make("ALE/Boxing-v5", render_mode="rgb_array", full_action_space=False)
    return env

# محیط برداری برای عامل RL
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

# محیط جداگانه برای رندر انسانی
render_env = make_env()

obs = env.reset()
render_env.reset()  # ✅ رفع خطای ResetNeeded

# ------------------------------
# راه‌اندازی pygame
# ------------------------------
pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Boxing: Human vs Agent")
clock = pygame.time.Clock()

# ------------------------------
# نگاشت کلیدها به اکشن‌های انسانی
# ------------------------------
key_action_map = {
    pygame.K_LEFT: 3,
    pygame.K_RIGHT: 2,
    pygame.K_a: 12,
    pygame.K_s: 13,
}

human_action = 0
done = False

# ------------------------------
# حلقه بازی
# ------------------------------
while not done:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key in key_action_map:
                human_action = key_action_map[event.key]
        elif event.type == pygame.KEYUP:
            human_action = 0

    # پیش‌بینی اکشن عامل RL
    agent_action, _ = model.predict(obs, deterministic=True)
    agent_action = int(agent_action[0])

    # ترکیب اکشن‌ها: [عامل, انسان]
    actions = [agent_action, int(human_action)]
    obs, reward, done, info = env.step(actions)
    done = done[0]

    # اعمال فقط اکشن انسانی روی محیط رندر
    render_env.step(human_action)
    raw_frame = render_env.render()

    # تبدیل تصویر به سطح قابل نمایش
    if raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
        frame = np.transpose(raw_frame, (1, 0, 2))  # تبدیل به (عرض، ارتفاع، رنگ)
    else:
        frame = raw_frame

    surface = pygame.surfarray.make_surface(frame)
    scaled_surface = pygame.transform.scale(surface, (screen_width, screen_height))
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()
    clock.tick(60)

# ------------------------------
# پاکسازی
# ------------------------------
env.close()
render_env.close()
pygame.quit()
