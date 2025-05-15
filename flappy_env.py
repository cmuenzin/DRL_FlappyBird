import gym
from gym import spaces
import numpy as np
import pygame
import sys

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()

        # Action space: 0 = no flap, 1 = flap
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

        # Pygame window setup
        self.window_width = 288
        self.window_height = 512
        self.bird_x = 50
        self.pipe_width = 50
        self.pipe_gap = 100

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Flappy Bird RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

    def reset(self):
        self.bird_y = 250
        self.velocity = 0
        self.pipe_x = self.window_width
        self.pipe_y = np.random.randint(100, 400)
        self.done = False
        return self._get_obs()

    def step(self, action):
        reward = 0.1

        if action == 1:
            self.velocity = -9
        else:
            self.velocity += 1
        self.bird_y += self.velocity
        self.pipe_x -= 5

        if self.pipe_x < -self.pipe_width:
            reward += 1
            self.pipe_x = self.window_width
            self.pipe_y = np.random.randint(100, 400)

        if (self.bird_y < 0 or self.bird_y > self.window_height or
                not (self.pipe_y - self.pipe_gap / 2 < self.bird_y < self.pipe_y + self.pipe_gap / 2)
                and self.pipe_x < self.bird_x + 25 < self.pipe_x + self.pipe_width):
            self.done = True
            reward = -100

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        return np.array([self.bird_y, self.velocity, self.pipe_x, self.pipe_y], dtype=np.float32)

    def render(self, mode="human"):
        self.screen.fill((135, 206, 235))  # Himmelblau

        # Bird
        pygame.draw.circle(self.screen, (255, 255, 0), (self.bird_x, int(self.bird_y)), 10)

        # Pipe
        pipe_color = (34, 139, 34)
        pygame.draw.rect(self.screen, pipe_color, pygame.Rect(
            self.pipe_x, 0, self.pipe_width, self.pipe_y - self.pipe_gap // 2))
        pygame.draw.rect(self.screen, pipe_color, pygame.Rect(
            self.pipe_x, self.pipe_y + self.pipe_gap // 2, self.pipe_width, self.window_height))

        # Info Text
        info_text = self.font.render(f'Y: {int(self.bird_y)} | V: {int(self.velocity)}', True, (0, 0, 0))
        self.screen.blit(info_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
