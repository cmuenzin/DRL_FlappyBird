from stable_baselines3 import PPO
from flappy_env import FlappyBirdEnv

env = FlappyBirdEnv()
model = PPO.load("flappy_ppo")

obs = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()
