import numpy as np
import pygame
import sys

class Renderer:
    def __init__(self, sim, width=800, height=800, scale=60):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("RL Omnidirectional + A* Planner")
        self.clock = pygame.time.Clock()
        self.sim = sim
        
        self.width = width
        self.height = height
        self.scale = scale # Pixels / meter
        
        # colors
        self.BG_COLOR = (20, 20, 20)
        self.WALL_COLOR = (200, 200, 200)
        self.ROBOT_COLOR = (0, 255, 0)
        self.OBJ_COLOR = (255, 0, 0)
        self.RAY_COLOR = (255, 50, 50)
        self.WAYPOINT_COLOR = (0, 150, 255)

    def to_screen(self, x, y):
        screen_x = int(self.width / 2 + x * self.scale)
        screen_y = int(self.height / 2 - y * self.scale)
        return (screen_x, screen_y)

    def run(self):
        running = True
        
        while running:
            vx = 0.0
            vy = 0.0
            omega = 0.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]:    vy = 1.0
            if keys[pygame.K_DOWN]:  vy = -1.0
            if keys[pygame.K_RIGHT]: vx = 1.0
            if keys[pygame.K_LEFT]:  vx = -1.0
            

            if keys[pygame.K_q]: omega = 2.0  
            if keys[pygame.K_e]: omega = -2.0 

            self.sim.step(vx, vy, omega)
            
            distances, angles = self.sim.read_sensors()

            self.screen.fill(self.BG_COLOR)
            
            for wall in self.sim.walls:
                start_pos = self.to_screen(wall[0], wall[1])
                end_pos = self.to_screen(wall[2], wall[3])
                pygame.draw.line(self.screen, self.WALL_COLOR, start_pos, end_pos, 3)
               
            rx, ry, _ = self.sim.pose
            start_ray = self.to_screen(rx, ry)
            for dist, angle in zip(distances, angles):
                end_x = rx + dist * np.cos(angle)
                end_y = ry + dist * np.sin(angle)
                end_ray = self.to_screen(end_x, end_y)
                pygame.draw.line(self.screen, self.RAY_COLOR, start_ray, end_ray, 1)
            
            objective = self.to_screen(self.sim.obj_x, self.sim.obj_y)
            pygame.draw.circle(self.screen, self.ROBOT_COLOR, start_ray, int(0.2 * self.scale))
            pygame.draw.circle(self.screen, self.OBJ_COLOR, objective, int(0.15 * self.scale))
            
            pygame.display.flip()
            
            self.clock.tick(60)

        pygame.quit()
        sys.exit()