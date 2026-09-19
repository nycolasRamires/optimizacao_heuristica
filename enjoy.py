import argparse
import sys
import numpy as np
import pygame
from stable_baselines3 import SAC
from rl_training import OmniDroneEnv

def main():
    parser = argparse.ArgumentParser(description="Visualize SAC Agent in Maze")
    parser.add_argument("--model", type=str, default="base", help="Model checkpoint path (default: 'base')")
    parser.add_argument("--arena_size", type=float, default=10.0, help="Arena size in meters (default: 10.0)")
    args = parser.parse_args()

    # Load environment and trained model
    env = OmniDroneEnv(arena_size=args.arena_size)
    print(f"Loading model '{args.model}'...")
    model = SAC.load(args.model, env=env)

    # Pygame frontend init
    pygame.init()
    width, height = 800, 800
    scale = int((min(width, height) * 0.88) / env.sim.arena_size)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"SAC Drone Inference - {args.model}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    def to_screen(x, y):
        return int(width / 2 + x * scale), int(height / 2 - y * scale)

    obs, info = env.reset()
    episodes = 0
    ep_reward = 0.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    ep_reward = 0.0

        # Predict deterministic action from policy
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        # Render frame
        screen.fill((20, 20, 25))
        
        # Draw Pillars (shaded fill + crisp border)
        for p in env.sim.pillars:
            top_left = to_screen(p["x_min"], p["y_max"])
            w_px = int((p["x_max"] - p["x_min"]) * scale)
            h_px = int((p["y_max"] - p["y_min"]) * scale)
            rect = pygame.Rect(top_left[0], top_left[1], w_px, h_px)
            pygame.draw.rect(screen, (55, 55, 70), rect)
            pygame.draw.rect(screen, (140, 140, 160), rect, 2)

        # Draw outer walls
        for wall in env.sim.walls[:4]:
            pygame.draw.line(screen, (210, 210, 210), to_screen(wall[0], wall[1]), to_screen(wall[2], wall[3]), 2)

        # Draw LiDAR rays
        rx, ry, rtheta = env.sim.pose
        start_ray = to_screen(rx, ry)
        distances, angles = env.sim.read_sensors()
        for dist, angle in zip(distances, angles):
            end_x = rx + dist * np.cos(angle)
            end_y = ry + dist * np.sin(angle)
            pygame.draw.line(screen, (180, 40, 40), start_ray, to_screen(end_x, end_y), 1)

        # Draw Drone (green circle + heading indicator)
        pygame.draw.circle(screen, (0, 255, 120), start_ray, max(3, int(env.sim.robot_radius * scale)))
        hx = rx + env.sim.robot_radius * 1.8 * np.cos(rtheta)
        hy = ry + env.sim.robot_radius * 1.8 * np.sin(rtheta)
        pygame.draw.line(screen, (255, 255, 0), start_ray, to_screen(hx, hy), 2)

        # Draw Objective
        pygame.draw.circle(screen, (255, 60, 60), to_screen(env.sim.obj_x, env.sim.obj_y), max(3, int(0.2 * scale)))

        # HUD text
        hud = font.render(f"Ep: {episodes} | Step: {env.current_step} | Spd: {env.sim.speed:.2f}m/s | Reward: {ep_reward:.1f} | [R] Reset", True, (180, 180, 180))
        screen.blit(hud, (15, 15))

        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            episodes += 1
            ep_reward = 0.0
            obs, info = env.reset()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()