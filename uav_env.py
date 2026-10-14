from collections import deque
import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, ActionType, Physics

import pybullet as p
from gymnasium import spaces

# Order: computeObs -> _computeReward -> _computeTerminated -> _computeTruncated
class Drone3DEnv(BaseRLAviary):
    def __init__(self, drone_model=DroneModel.CF2X, 
                 initial_xyzs=np.array([[0.0, 0.0, 1.0]]), 
                 initial_rpys=None, 
                 physics=Physics.PYB, 
                 pyb_freq=200, ctrl_freq=40, 
                 gui=False, record=False, 
                 act = ActionType.PID,
                 output_folder='results'):

        self.goal_pos = np.array([0.0, 0.0, 1.0])
        self.obstacle_ids = []
        # Obstacle parameters: 8 thick cylindrical pillars
        self.num_obstacles = 8
        self.obstacle_radius = 0.20

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            act=act
        )

        self.max_steps = 500
        self.current_step = 0

        self.cam_w = 64
        self.cam_h = 64
        self.fov = 90.0
        self.near = 0.1
        self.far = 10.0
        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.cam_w / self.cam_h, self.near, self.far)

        # Observation space: 64x64 depth image and 23D kinematics state
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(1, self.cam_h, self.cam_w), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32),
        })
        
        self.min_sensor_dist = 10.0
        self.last_dist_to_goal = 0.0

        # Multi-step retreat tolerance buffer to navigate around obstacles
        self.recent_distances = deque(maxlen=10)
        self.retreat_tolerance = 0.10
        self.retreat_multiplier = 2.0

    def _actionSpace(self):
        # Override default ±1m destination cap of BaseRLAviary PID mode
        # Treats action as absolute world coordinates destination
        act_lower = np.array([[-2.5, -2.5, 0.0]])
        act_upper = np.array([[ 2.5,  2.5, 2.5]])
        return spaces.Box(low=act_lower, high=act_upper, dtype=np.float32)

    def step(self, action):
        # Reshape 1D action to (NUM_DRONES, -1) for compatibility with vectorized envs
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(self.NUM_DRONES, -1)
        return super().step(action)

    def _computeInfo(self):
        # Return episode outcome and spatial metrics for telemetry tracking
        pos = self.state[0:3]
        dist = float(np.linalg.norm(self.goal_pos - pos))
        return {
            "dist_to_goal": dist,
            "is_success": dist < 0.2,
            "collision": self._checkPhysicalCollision() or pos[2] < 0.1,
            "timeout": self.current_step >= self.max_steps,
        }

    def reset(self, seed=None, options=None):
        self.current_step = 0

        self.goal_pos = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(0.5, 2.0)
        ])

        _, info = super().reset(seed=seed, options=options)

        self.state = self._getDroneStateVector(0)
        pos = self.state[0:3]
        self.last_dist_to_goal = float(np.linalg.norm(self.goal_pos - pos))

        # Reset retreat tolerance distance buffer with initial goal distance
        self.recent_distances.clear()
        self.recent_distances.append(self.last_dist_to_goal)

        initial_obs = self._computeObs()

        return initial_obs, info

    def _computeRewardHoverStage(self):
        # Unused reward policy for hovering in place
        pos = self.state[0:3]
        rpy = self.state[7:10]
        vel = self.state[10:13]
        
        if pos[2] < 0.1:
            return -100.0
            
        survival_bonus = 5.0
        altitude_reward = 1.0 - abs(1.0 - pos[2])
        
        velocity_penalty = np.linalg.norm(vel) * 0.5 
        drift_penalty = np.linalg.norm(pos[0:2]) * 0.2
        acrobatics_penalty = (abs(rpy[0]) + abs(rpy[1])) * 2.0
        
        return float(survival_bonus + altitude_reward - velocity_penalty - drift_penalty - acrobatics_penalty)

    def _computeReward(self):
        pos = self.state[0:3]
        quat = self.state[3:7]
        vel = self.state[10:13]

        reward = 0.0

        # Cost of existing per timestep to incentivize rapid goal reaching
        reward -= 0.01

        # Ground collision penalty
        if pos[2] < 0.1:
            reward -= 50.0

        # Physical collision penalty with obstacles or ground plane
        if self._checkPhysicalCollision():
            reward -= 50.0

        # Proximity buffer penalty when near obstacles
        if self.min_sensor_dist < 0.5:
            reward -= (0.5 - self.min_sensor_dist) * 2.0

        # Velocity alignment penalty for flying outside camera FOV
        vel_norm = np.linalg.norm(vel)
        if vel_norm > 0.1:
            rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            forward_vec = rot_matrix[:, 0]
            vel_dir = vel / vel_norm
            alignment = np.dot(forward_vec, vel_dir)
            fov_threshold = np.cos(np.deg2rad(self.fov / 2.0))

            if alignment < fov_threshold: 
                reward -= (fov_threshold - alignment) * 1.0

        # Potential-based progress shaping reward towards goal
        dist_to_goal = float(np.linalg.norm(self.goal_pos - pos))
        progress = self.last_dist_to_goal - dist_to_goal
        reward += progress * 10.0
        self.last_dist_to_goal = dist_to_goal

        # Multi-step retreat tolerance penalty to prevent obstacle deadlock
        if len(self.recent_distances) > 0:
            retreat_dist = dist_to_goal - min(self.recent_distances)
            if retreat_dist > self.retreat_tolerance:
                reward -= (retreat_dist - self.retreat_tolerance) * self.retreat_multiplier
        self.recent_distances.append(dist_to_goal)

        # Terminal success bonus upon reaching goal radius
        if dist_to_goal < 0.2:
            reward += 100.0

        return float(reward)

    def _checkPhysicalCollision(self):
        # Ground-truth collision check via PyBullet contact manifold
        # Catches obstacle or ground hits regardless of camera view
        drone_id = self.DRONE_IDS[0]
        contacts = p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT)

        if len(contacts) == 0:
            return False

        contact_body_ids = {c[2] for c in contacts}
        relevant_ids = set(getattr(self, 'obstacle_ids', [])) | {getattr(self, 'PLANE_ID', -1)}

        return bool(contact_body_ids & relevant_ids)

    def _computeTerminated(self):
        pos = self.state[0:3]

        # finish if ground hit
        if pos[2] < 0.1:
            return True
        
        if np.linalg.norm(self.goal_pos - pos) < 0.2:
            return True

        if self._checkPhysicalCollision():
            return True
        
        return False

    def _computeObs(self):
        # Kinematics state and forward depth camera observation constructor
        self.state = self._getDroneStateVector(0)
        safe_state = self.state.copy()
        safe_state[10:13] = np.clip(safe_state[10:13], -10.0, 10.0)
        safe_state[13:16] = np.clip(safe_state[13:16], -20.0, 20.0)

        pos = self.state[0:3]
        quat = self.state[3:7]
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Rotate relative goal vector from world into body frame
        rel_goal_world = (self.goal_pos - pos).astype(np.float32)
        rel_goal_body = rot_matrix.T.dot(rel_goal_world).astype(np.float32)
        kin_obs = np.concatenate([safe_state, rel_goal_body]).astype(np.float32)

        # Forward and up vectors from orientation matrix columns
        forward_vec = rot_matrix[:, 0]
        up_vec = rot_matrix[:, 2]
        target_pos = pos + forward_vec

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vec
        )
        _renderer = p.ER_TINY_RENDERER if self.GUI else p.ER_BULLET_HARDWARE_OPENGL
        # Render depth camera with segmentation disabled for speed
        _, _, _, depth_img, _ = p.getCameraImage(
            width=self.cam_w, height=self.cam_h,
            viewMatrix=view_matrix, projectionMatrix=self.proj_matrix,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=_renderer
        )
        depth_array = np.array(depth_img, dtype=np.float32).reshape(self.cam_h, self.cam_w, 1)
        depth_linear = self.far * self.near / (self.far - (self.far - self.near) * depth_array)
        self.min_sensor_dist = float(np.min(depth_linear))

        # Linear metric depth normalized to [0, 1] then scaled to uint8 [0, 255]
        norm_depth = np.clip((depth_linear - self.near) / (self.far - self.near), 0.0, 1.0)
        image_obs = (norm_depth * 255.0).astype(np.uint8).transpose(2, 0, 1)  # → (1, H, W)

        return {"image": image_obs, "state": kin_obs}

    def _addObstacles(self):
        # Spawns floor
        super()._addObstacles()
        # spawns the pillars
        self._addObstacles2()
    
    def _addObstacles2(self):
        self.obstacle_ids = []

        # OBJECTIVE (Green Sphere) 
        # Creates a visible body without a collision hitbox so the drone can fly into it
        goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[0.0, 1.0, 0.0, 0.8])
        self.goal_id = p.createMultiBody(
            baseMass=0, # Static object
            baseCollisionShapeIndex=-1, # No collision physics
            baseVisualShapeIndex=goal_vis,
            basePosition=self.goal_pos,
            physicsClientId=self.CLIENT
        )

        # Spawn cylindrical obstacle pillars with pairwise and goal clearance
        drone_xy = np.array(self.INIT_XYZS[0][0:2]) if self.INIT_XYZS is not None else np.array([0.0, 0.0])
        goal_xy = np.array(self.goal_pos[0:2])

        min_clearance = 0.55
        placed_pillars = []

        for _ in range(self.num_obstacles):
            candidate_xy = None
            for _attempt in range(50):
                c = np.array([
                    np.random.uniform(-1.5, 1.5),
                    np.random.uniform(-1.5, 1.5)
                ])
                clears_drone = np.linalg.norm(c - drone_xy) >= min_clearance
                clears_goal = np.linalg.norm(c - goal_xy) >= min_clearance
                clears_pillars = not placed_pillars or all(np.linalg.norm(c - p) >= min_clearance for p in placed_pillars)
                if clears_drone and clears_goal and clears_pillars:
                    candidate_xy = c
                    break

            if candidate_xy is None:
                candidate_xy = np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)])

            placed_pillars.append(candidate_xy)
            pos = [candidate_xy[0], candidate_xy[1], 1.0]

            col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.obstacle_radius, height=2.0)
            vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=self.obstacle_radius, length=2.0, rgbaColor=[0.7, 0.2, 0.2, 1])

            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=pos,
                physicsClientId=self.CLIENT
            )
            self.obstacle_ids.append(body_id)

    def _computeTruncated(self):
        #ending the training by timeout (sometime the drone gets stuck, necessary)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            return True
        return False