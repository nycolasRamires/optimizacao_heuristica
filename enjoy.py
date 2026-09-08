import pygame
import sys
import numpy as np
from stable_baselines3 import SAC
from rl_training import OmniDroneEnv

# load env
env = OmniDroneEnv()
model = SAC.load("sac_drone_v7") #load model

# Frontend init
pygame.init()
width, height = 800, 800
scale = 60
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("SAC Agent Inference")
clock = pygame.time.Clock()

def to_screen(x, y):
    """Utilitário para converter coordenadas"""
    return int(width / 2 + x * scale), int(height / 2 - y * scale)

obs, info = env.reset()

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # neural network actions
    action, _states = model.predict(obs, deterministic=True)

    # aplies vx, vy and omega and gets new state
    obs, reward, terminated, truncated, info = env.step(action)

    # if scenario terminated
    if terminated or truncated:
        obs, info = env.reset()

    # render
    screen.fill((20, 20, 20))
    
    # draw walls
    for wall in env.sim.walls:
        pygame.draw.line(screen, (200, 200, 200), to_screen(wall[0], wall[1]), to_screen(wall[2], wall[3]), 3)

    # LiDAR
    rx, ry, _ = env.sim.pose
    start_ray = to_screen(rx, ry)
    distances, angles = env.sim.read_sensors()
    for dist, angle in zip(distances, angles):
        end_x = rx + dist * np.cos(angle)
        end_y = ry + dist * np.sin(angle)
        pygame.draw.line(screen, (255, 50, 50), start_ray, to_screen(end_x, end_y), 1)

    # UAV
    pygame.draw.circle(screen, (0, 255, 0), start_ray, int(0.2 * scale))
    pygame.draw.circle(screen, (255, 0, 0), to_screen(env.sim.obj_x, env.sim.obj_y), int(0.15 * scale))

    # updates screen
    pygame.display.flip()
    
    # tries to set fps to 60
    clock.tick(60)

pygame.quit()
sys.exit()