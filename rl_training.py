import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simple_sim import MazeSimulation # physics engine


class OmniDroneEnv(gym.Env):
    """
    Wrapper do Gymnasium para treinar o Drone Omnidirecional com A*.
    """
    def __init__(self):
        super().__init__()
        
        # sim engine
        self.sim = MazeSimulation(dt=1.0/60.0)
        
        # robot parameters
        self.max_v = 2.0     # m/s
        self.max_omega = 3.0 # rad/s
        self.max_steps = 500 # Timeout
        
        # action space: vx, vy, omega (norm -1  1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # inputs: 15 laser (lidar) + dist to waypoint + angle to waypoint 
        # normalized 0 to 1 (ou -1 to 1 for angles)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        # resets drone pose to center
        self.sim.pose = np.array([0.0, -3.0, 0.0])
        #new objective
        self.sim.gen_rand_obj()
        
        self.last_dist_to_goal = self._get_dist_to_current_wp()
        
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        
        # undo normalizaton
        vx = action[0] * self.max_v
        vy = action[1] * self.max_v
        omega = action[2] * self.max_omega
        
        # run a step
        self.sim.step(vx, vy, omega)
        
        # Rewards
        reward = -0.01 # living cost 
        terminated = False
        truncated = False
        
        # check colision
        dist_wall = self.sim._get_min_dist_to_walls(self.sim.pose[0], self.sim.pose[1])

        if dist_wall <= self.sim.robot_radius + 0.01:
            reward -= 1.0
            terminated = True
            return self._get_observation(), reward, terminated, truncated, {}

        if dist_wall <= self.sim.robot_radius + 0.075: # hit the wall
            reward -= 0.05


        # Calculate angle to goal locally
        rx, ry, rtheta = self.sim.pose
        dx, dy = self.sim.obj_x - rx, self.sim.obj_y - ry
        local_x = dx * np.cos(-rtheta) - dy * np.sin(-rtheta)
        local_y = dx * np.sin(-rtheta) + dy * np.cos(-rtheta)
        angle_to_goal = np.arctan2(local_y, local_x)

        # Direction multiplier: 1.0 when facing perfectly, dropping to 0.1 if facing away.
        # This prevents negative rewards for moving closer while facing backwards, 
        # but massively incentivizes moving forward.
        direction_factor = max(0.1, np.cos(angle_to_goal))

        # current Waypoint dist logic
        current_dist = self._get_dist_to_current_wp()
        reward += (self.last_dist_to_goal - current_dist) * 10.0 * direction_factor

        if abs(self.last_dist_to_goal - current_dist) < self.sim.robot_radius/15:
            reward -= 0.01
        self.last_dist_to_goal = current_dist
        
        # Tax for loitering / freezing
        if abs(vx) + abs(vy) + abs(omega) < 0.1:
            reward -= 0.1

        if current_dist < 0.25:
            reward += 100.0
            self.sim.gen_rand_obj() # got the waypoint
            terminated = True

        if self.sim.is_wp_in_fov():
            reward += 0.01

        # Timeout
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_dist_to_current_wp(self):
        ox=self.sim.obj_x
        oy=self.sim.obj_y

        return np.linalg.norm([self.sim.pose[0] - ox, self.sim.pose[1] - oy])

    def _get_observation(self):
        obs = np.zeros(17, dtype=np.float32)
        
        # LiDAR (15 normalized values 0 to 1)
        distances, _ = self.sim.read_sensors()
        obs[0:15] = np.clip(distances / self.sim.max_range, 0.0, 1.0)

                
        # Waypoint location
        ox=self.sim.obj_x
        oy=self.sim.obj_y

        rx, ry, rtheta = self.sim.pose
        
        # abs
        dx = ox - rx
        dy = oy - ry
        
        # Converts to robot reference
        local_x = dx * np.cos(-rtheta) - dy * np.sin(-rtheta)
        local_y = dx * np.sin(-rtheta) + dy * np.cos(-rtheta)
        
        dist = np.linalg.norm([local_x, local_y])
        angle = np.arctan2(local_y, local_x)
        
        # Normalize assuming 10x10 maze (diagonal ~14)
        obs[15] = np.clip(dist / 14.0, 0.0, 1.0)
        obs[16] = angle / np.pi # Entre -1 e 1
        
        return obs