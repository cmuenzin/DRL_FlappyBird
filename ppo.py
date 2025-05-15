from stable_baselines3 import PPO
from flappy_env import FlappyBirdEnv

env = FlappyBirdEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Optional: speichern
model.save("flappy_ppo")
