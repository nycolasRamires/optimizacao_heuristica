import numpy as np
import heapq
from Render import *

class MazeSimulation:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.robot_radius = 0.2
        
        # Pose: [x, y, theta]
        self.pose = np.array([0.0, -3.0, np.pi/2])
        
        # walls
        self.walls = np.array([
            # Outer Bounds
            [-5.0, -5.0,  5.0, -5.0], 
            [ 5.0, -5.0,  5.0,  5.0], 
            [ 5.0,  5.0, -5.0,  5.0], 
            [-5.0,  5.0, -5.0, -5.0], 
            
            # Single central barrier
            [-2.0,  0.0,  2.0,  0.0] 
        ])
        
        # Sensor init
        self.num_rays = 15
        self.fov = np.deg2rad(100)
        self.max_range = 10.0

        # target
        self.obj_x, self.obj_y = 0.0, 0.0
        
        self.gen_rand_obj()

    def step(self, vx, vy, omega):
        """
        Inputs: vx, vy, omega.
        vx: vel X global
        vy: vel Y global
        """
        old_x, old_y, old_theta = self.pose
        
        # Convert local agent velocities to global map velocities
        global_vx = vx * np.cos(old_theta) - vy * np.sin(old_theta)
        global_vy = vx * np.sin(old_theta) + vy * np.cos(old_theta)
        
        # cinematic 
        new_x = old_x + global_vx * self.dt
        new_y = old_y + global_vy * self.dt
        new_theta = old_theta + omega * self.dt
        
        # theta normalization
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        
        # dist to wall
        dist_parede = self._get_min_dist_to_walls(new_x, new_y)
        
        if dist_parede >= self.robot_radius:
            self.pose = np.array([new_x, new_y, new_theta])
        else:
            self.pose = np.array([old_x, old_y, new_theta])


    def is_wp_in_fov(self):
        x, y, theta = self.pose
        v = np.array([1,0])
        v1 = self._rotate_vector(theta+self.fov/2, v)
        v2 = self._rotate_vector(theta-self.fov/2, v)
        sight = np.column_stack((v1, v2))

        obj = np.array([self.obj_x-x,self.obj_y-y])

        r = np.linalg.solve(sight,obj)
        return bool(np.all(r >= 0))

    
    def _rotate_vector(self, theta, v):
        R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        return R.dot(v)
    
    def _get_min_dist_to_walls(self, x, y):
        P = np.array([x, y])
        A = self.walls[:, 0:2]
        B = self.walls[:, 2:4]
        AB = B - A
        dot_AB_AB = np.sum(AB * AB, axis=1) + 1e-8 
        AP = P - A
        dot_AP_AB = np.sum(AP * AB, axis=1)
        t = np.clip(dot_AP_AB / dot_AB_AB, 0.0, 1.0)
        closest_points = A + t[:, np.newaxis] * AB
        distances = np.linalg.norm(P - closest_points, axis=1)
        return np.min(distances)

    def read_sensors(self):
        pose_x, pose_y, pose_theta = self.pose
        angles = np.linspace(pose_theta - self.fov/2, pose_theta + self.fov/2, self.num_rays)
        distances = np.full(self.num_rays, self.max_range)
        
        ray_origins = np.array([pose_x, pose_y])
        ray_dirs = np.column_stack((np.cos(angles), np.sin(angles)))
        
        for wall in self.walls:
            A = wall[0:2]
            B = wall[2:4]
            wall_vec = B - A
            
            for i in range(self.num_rays):
                ray_vec = ray_dirs[i]
                det = ray_vec[0] * wall_vec[1] - ray_vec[1] * wall_vec[0]
                if abs(det) < 1e-6:
                    continue
                diff = A - ray_origins
                t = (diff[0] * wall_vec[1] - diff[1] * wall_vec[0]) / det
                u = (diff[0] * ray_vec[1] - diff[1] * ray_vec[0]) / det
                if t > 0 and 0 <= u <= 1:
                    if t < distances[i]:
                        distances[i] = t
        return distances, angles

    def gen_rand_obj(self, min_dist=0.6):
        x_min, x_max = -4.5, 4.5
        y_min, y_max = -4.5, 4.5
        
        while True:
            c_x = np.random.uniform(x_min, x_max)
            c_y = np.random.uniform(y_min, y_max)
            if self._get_min_dist_to_walls(c_x, c_y) > min_dist:
                self.obj_x = c_x
                self.obj_y = c_y
                break
        

if __name__ == "__main__":
    sim = MazeSimulation()
    # ui = Renderer(sim)
    # ui.run()