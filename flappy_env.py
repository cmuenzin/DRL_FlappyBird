import gym
from gym import spaces
import numpy as np

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        
        # Actions: 0 = no flap, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Observation: bird y, velocity, next pipe x/y
        low = np.array([0, -10, 0, 0], dtype=np.float32)
        high = np.array([512, 10, 288, 512], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Internals
        self.bird_y = 0
        self.velocity = 0
        self.pipe_x = 0
        self.pipe_y = 0
        self.done = False

    def reset(self):
        # Reset state
        self.bird_y = 250
        self.velocity = 0
        self.pipe_x = 288
        self.pipe_y = np.random.randint(100, 400)
        self.done = False
        return self._get_obs()

    def step(self, action):
        reward = 0.1  # small living reward
        if action == 1:
            self.velocity = -9
        else:
            self.velocity += 1
        self.bird_y += self.velocity
        self.pipe_x -= 5

        # Reward for passing a pipe
        if self.pipe_x < 0:
            reward += 1
            self.pipe_x = 288
            self.pipe_y = np.random.randint(100, 400)

        # Check collision
        if self.bird_y < 0 or self.bird_y > 512 or not (self.pipe_y - 50 < self.bird_y < self.pipe_y + 50):
            self.done = True
            reward = -100

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        return np.array([self.bird_y, self.velocity, self.pipe_x, self.pipe_y], dtype=np.float32)

    def render(self, mode="human"):
        pass  # für später

