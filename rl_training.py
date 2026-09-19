import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections
from simple_sim import MazeSimulation

DEFAULT_REWARD_PARAMS = {
    "living_cost": 0.01,
    "collision_penalty": 1.0,
    "crash_speed_penalty": 2.0,            # Extra penalty scaling with impact speed
    "high_speed_crash_penalty": 5.0,       # Severe penalty for crashes above safe velocity threshold
    "wall_proximity_penalty": 0.05,
    "progress_multiplier": 10.0,
    "stagnation_penalty": 0.01,
    "loitering_penalty": 0.1,
    "high_speed_reward": 0.02,             # Reward for flying at high velocity
    "overspeed_penalty": 0.05,             # Penalty for exceeding safe velocity threshold (1.2 m/s)
    "constant_vel_reward": 0.02,           # Reward for cruising at steady, non-jerky velocity
    "goal_reward": 100.0,
    "early_goal_multiplier": 0.1,
    "fov_bonus": 0.01,
    "direction_factor_min": 0.1,
    "retreat_tolerance": 0.05,
    "retreat_penalty": 0.02,
    "retreat_multiplier": 2.0,
    "retreat_window_steps": 10
}

class OmniDroneEnv(gym.Env):
    """
    Gymnasium wrapper for the 2D Omnidirectional Drone.
    Optimized for high-throughput vectorized simulation, expanded mazes with pillars,
    acceleration/deceleration control, and genetic reward tuning.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, reward_params=None, arena_size=10.0):
        super().__init__()
        
        # Physics engine (60 Hz, 10x10m compact arena with 3 pillars)
        self.sim = MazeSimulation(dt=1.0/60.0, arena_size=arena_size)
        
        # Robot physical limits (acceleration and physical max velocity)
        self.max_a = 3.0      # m/s^2 (max acceleration/deceleration)
        self.max_v = 3.0      # m/s (physical max absolute velocity)
        self.max_omega = 3.0  # rad/s (angular speed)
        self.max_steps = 500  # Timeout
        
        # Configurable reward parameters (for genetic algorithm optimization)
        self.reward_params = DEFAULT_REWARD_PARAMS.copy()
        if reward_params:
            self.reward_params.update(reward_params)
        
        # Action space: ax, ay (accelerations), omega (angular rate) normalized [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 15 LiDAR rays + goal dist + goal angle + local vx + local vy (19D)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
        
        # Pre-allocated buffer for zero-overhead observations
        self.obs_buf = np.zeros(19, dtype=np.float32)
        self.inv_max_range = 1.0 / self.sim.max_range
        self.inv_diag = 1.0 / self.sim.arena_diagonal
        self.inv_pi = 1.0 / np.pi
        self.stagnation_threshold = self.sim.robot_radius / 15.0

        # Rolling history buffer for retreat penalty with tolerance
        window_size = max(1, int(self.reward_params.get("retreat_window_steps", 10)))
        self.dist_history = collections.deque(maxlen=window_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Spawn drone at a new randomized valid location and heading
        self.sim.gen_valid_pose()
        
        # Spawn target at a new randomized valid location clear of pillars & walls
        self.sim.gen_rand_obj(min_dist_to_robot=2.0)
        self.last_dist_to_goal = self._get_dist_to_current_wp()
        
        # Reset retreat history
        window_size = max(1, int(self.reward_params.get("retreat_window_steps", 10)))
        if self.dist_history.maxlen != window_size:
            self.dist_history = collections.deque(maxlen=window_size)
        self.dist_history.clear()
        self.dist_history.append(self.last_dist_to_goal)
        
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        
        # Unnormalize action: accelerations ax, ay in local body frame, angular speed omega
        ax = float(action[0]) * self.max_a
        ay = float(action[1]) * self.max_a
        omega = float(action[2]) * self.max_omega
        
        # Previous speed before kinematic step
        prev_speed = self.sim.speed
        
        # Step simulation kinematics (2nd-order acceleration/deceleration update)
        self.sim.step(ax, ay, omega)
        
        current_speed = self.sim.speed
        
        rp = self.reward_params
        reward = -rp["living_cost"]
        terminated = False
        truncated = False
        
        # Check collision with outer walls or pillar boundaries
        dist_wall = self.sim.last_min_wall_dist
        inside_pillar = self.sim._is_inside_any_pillar(self.sim.pose[0], self.sim.pose[1])

        # Collision with wall or pillar (speed-dependent crash penalty)
        if dist_wall <= self.sim.robot_radius + 0.01 or inside_pillar:
            impact_spd = self.sim.impact_speed if self.sim.impact_speed > 0.0 else current_speed
            coll_penalty = rp["collision_penalty"]
            # Penalty for hitting wall in high speed
            coll_penalty += rp["crash_speed_penalty"] * (impact_spd / self.max_v)
            # Bigger if above safety threshold
            if impact_spd > self.sim.safe_v:
                excess_ratio = (impact_spd - self.sim.safe_v) / (self.max_v - self.sim.safe_v)
                coll_penalty += rp["high_speed_crash_penalty"] * excess_ratio
            reward -= coll_penalty
            terminated = True
            return self._get_observation(), reward, terminated, truncated, {"collision": True, "is_success": False}

        # Proximity buffer warning
        if dist_wall <= self.sim.robot_radius + 0.075:
            reward -= rp["wall_proximity_penalty"]

        # Policy 1: Reward for high speed
        reward += rp["high_speed_reward"] * (current_speed / self.max_v)

        # Policy 2: Penalty for flying above safe threshold (threshold fixed at safe_v = 1.2 m/s)
        if current_speed > self.sim.safe_v:
            overspeed_ratio = (current_speed - self.sim.safe_v) / (self.max_v - self.sim.safe_v)
            reward -= rp["overspeed_penalty"] * overspeed_ratio

        # Policy 3: Reward for maintaining constant velocity when in cruise motion
        if current_speed > 0.2:
            speed_delta = abs(current_speed - prev_speed)
            max_possible_delta = self.max_a * self.sim.dt
            stability = max(0.0, 1.0 - (speed_delta / max_possible_delta))
            reward += rp["constant_vel_reward"] * stability

        # Calculate relative target vector in robot local frame
        rx, ry, rtheta = self.sim.pose
        dx = self.sim.obj_x - rx
        dy = self.sim.obj_y - ry
        cos_t = np.cos(-rtheta)
        sin_t = np.sin(-rtheta)
        local_x = dx * cos_t - dy * sin_t
        local_y = dx * sin_t + dy * cos_t
        
        current_dist = float(np.hypot(dx, dy))
        angle_to_goal = float(np.arctan2(local_y, local_x))

        # Direction multiplier: prioritizes moving toward target while facing it
        direction_factor = max(rp["direction_factor_min"], np.cos(angle_to_goal))
        reward += (self.last_dist_to_goal - current_dist) * rp["progress_multiplier"] * direction_factor

        # Penalty for getting away from the goal relative to recent steps with tolerance
        if len(self.dist_history) > 0:
            min_recent_dist = min(self.dist_history)
            dist_increase = current_dist - min_recent_dist
            if dist_increase > rp["retreat_tolerance"]:
                excess = dist_increase - rp["retreat_tolerance"]
                reward -= (rp["retreat_penalty"] + rp["retreat_multiplier"] * excess)
        self.dist_history.append(current_dist)

        # Stagnation penalty
        if abs(self.last_dist_to_goal - current_dist) < self.stagnation_threshold:
            reward -= rp["stagnation_penalty"]
        self.last_dist_to_goal = current_dist
        
        # Loitering / freezing penalty
        if current_speed + abs(omega) < 0.1:
            reward -= rp["loitering_penalty"]

        is_success = False
        # Goal reached
        if current_dist < 0.25:
            # Fixed goal reward
            reward += rp["goal_reward"]
            # Early arrival reward: less timesteps -> more points
            early_bonus = max(0, self.max_steps - self.current_step) * rp["early_goal_multiplier"]
            reward += early_bonus

            self.sim.gen_rand_obj(min_dist_to_robot=2.0)
            self.dist_history.clear()
            self.dist_history.append(self._get_dist_to_current_wp())
            terminated = True
            is_success = True

        # Field-of-view bonus
        if abs(angle_to_goal) <= self.sim.half_fov:
            reward += rp["fov_bonus"]

        # Episode timeout
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_observation(local_dist=current_dist, local_angle=angle_to_goal), reward, terminated, truncated, {"is_success": is_success, "collision": False}

    def _get_dist_to_current_wp(self):
        return float(np.hypot(self.sim.pose[0] - self.sim.obj_x, self.sim.pose[1] - self.sim.obj_y))

    def _get_observation(self, local_dist=None, local_angle=None):
        distances, _ = self.sim.read_sensors()
        self.obs_buf[0:15] = distances * self.inv_max_range
        
        if local_dist is None:
            rx, ry, rtheta = self.sim.pose
            dx = self.sim.obj_x - rx
            dy = self.sim.obj_y - ry
            cos_t = np.cos(-rtheta)
            sin_t = np.sin(-rtheta)
            local_x = dx * cos_t - dy * sin_t
            local_y = dx * sin_t + dy * cos_t
            local_dist = np.hypot(local_x, local_y)
            local_angle = np.arctan2(local_y, local_x)
            
        self.obs_buf[15] = np.clip(local_dist * self.inv_diag, 0.0, 1.0)
        self.obs_buf[16] = local_angle * self.inv_pi
        
        # Local body velocities normalized by physical max_v
        vx_local, vy_local = self.sim.get_local_velocity()
        self.obs_buf[17] = np.clip(vx_local / self.max_v, -1.0, 1.0)
        self.obs_buf[18] = np.clip(vy_local / self.max_v, -1.0, 1.0)
        
        return self.obs_buf.copy()