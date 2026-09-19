import numpy as np
import pygame
import sys

class Renderer:
    def __init__(self, sim, width=800, height=800, scale=None):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2D Omnidirectional Drone Arena")
        self.clock = pygame.time.Clock()
        self.sim = sim
        
        # Calculate scale to fit arena with margin
        arena_size = getattr(sim, "arena_size", 20.0)
        self.scale = scale if scale is not None else int((min(width, height) * 0.88) / arena_size)
        
        # Colors
        self.BG_COLOR = (20, 20, 25)
        self.WALL_COLOR = (210, 210, 210)
        self.PILLAR_FILL_COLOR = (55, 55, 70)
        self.PILLAR_EDGE_COLOR = (140, 140, 160)
        self.ROBOT_COLOR = (0, 255, 120)
        self.ROBOT_HEADING_COLOR = (255, 255, 0)
        self.OBJ_COLOR = (255, 60, 60)
        self.RAY_COLOR = (255, 50, 50, 100)

    def to_screen(self, x, y):
        screen_x = int(self.width / 2 + x * self.scale)
        screen_y = int(self.height / 2 - y * self.scale)
        return (screen_x, screen_y)

    def run(self):
        running = True
        font = pygame.font.SysFont(None, 24)
        
        while running:
            ax = 0.0
            ay = 0.0
            omega = 0.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.sim.gen_valid_pose()
                        self.sim.gen_rand_obj()
                    
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:    ax = 3.0   # Accelerate forward
            if keys[pygame.K_DOWN]:  ax = -3.0  # Decelerate / brake / reverse
            if keys[pygame.K_RIGHT]: ay = 3.0   # Accelerate lateral right
            if keys[pygame.K_LEFT]:  ay = -3.0  # Accelerate lateral left
            if keys[pygame.K_SPACE]:            # Active handbrake
                local_vx, local_vy = self.sim.get_local_velocity()
                ax = -np.sign(local_vx) * min(3.0, abs(local_vx) / self.sim.dt)
                ay = -np.sign(local_vy) * min(3.0, abs(local_vy) / self.sim.dt)
            if keys[pygame.K_q]:     omega = 2.0  
            if keys[pygame.K_e]:     omega = -2.0 

            self.sim.step(ax, ay, omega)
            distances, angles = self.sim.read_sensors()

            self.screen.fill(self.BG_COLOR)
            
            # Draw Pillars (shaded interiors)
            if hasattr(self.sim, "pillars"):
                for p in self.sim.pillars:
                    top_left = self.to_screen(p["x_min"], p["y_max"])
                    width_px = int((p["x_max"] - p["x_min"]) * self.scale)
                    height_px = int((p["y_max"] - p["y_min"]) * self.scale)
                    rect = pygame.Rect(top_left[0], top_left[1], width_px, height_px)
                    pygame.draw.rect(self.screen, self.PILLAR_FILL_COLOR, rect)
                    pygame.draw.rect(self.screen, self.PILLAR_EDGE_COLOR, rect, 2)
            
            # Draw Arena Walls
            for wall in self.sim.walls:
                start_pos = self.to_screen(wall[0], wall[1])
                end_pos = self.to_screen(wall[2], wall[3])
                pygame.draw.line(self.screen, self.WALL_COLOR, start_pos, end_pos, 2)
               
            # Draw LiDAR rays
            rx, ry, rtheta = self.sim.pose
            start_ray = self.to_screen(rx, ry)
            for dist, angle in zip(distances, angles):
                end_x = rx + dist * np.cos(angle)
                end_y = ry + dist * np.sin(angle)
                end_ray = self.to_screen(end_x, end_y)
                pygame.draw.line(self.screen, (180, 40, 40), start_ray, end_ray, 1)
            
            # Draw Drone
            pygame.draw.circle(self.screen, self.ROBOT_COLOR, start_ray, max(3, int(self.sim.robot_radius * self.scale)))
            # Heading indicator line
            heading_x = rx + self.sim.robot_radius * 1.8 * np.cos(rtheta)
            heading_y = ry + self.sim.robot_radius * 1.8 * np.sin(rtheta)
            pygame.draw.line(self.screen, self.ROBOT_HEADING_COLOR, start_ray, self.to_screen(heading_x, heading_y), 2)

            # Draw Objective
            objective = self.to_screen(self.sim.obj_x, self.sim.obj_y)
            pygame.draw.circle(self.screen, self.OBJ_COLOR, objective, max(3, int(0.2 * self.scale)))
            
            # HUD text
            hud = font.render(f"Pose: ({rx:.1f}, {ry:.1f}) | Spd: {self.sim.speed:.2f}m/s | Goal: ({self.sim.obj_x:.1f}, {self.sim.obj_y:.1f}) | [R] Respawn", True, (180, 180, 180))
            self.screen.blit(hud, (15, 15))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    from simple_sim import MazeSimulation
    sim = MazeSimulation(arena_size=10.0)
    Renderer(sim).run()