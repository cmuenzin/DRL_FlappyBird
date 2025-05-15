from stable_baselines3 import PPO
from flappy_env import FlappyBirdEnv

# Trainingsumgebung
env = FlappyBirdEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000000)

# Speichern
model.save("flappy_ppo")

# Evaluation: Agent spielt eine Weile
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        print("Game over. Resetting environment...\n")
        obs = env.reset()
