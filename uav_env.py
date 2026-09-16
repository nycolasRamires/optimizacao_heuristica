import numpy as np
# from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, ActionType, Physics

import pybullet as p
from gymnasium import spaces

# order computeObs -> _computeReward -> _computeTerminated -> _computeTruncated
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

        # defines the NN input
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(1, self.cam_h, self.cam_w), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32),  # was 20
        })
        
        self.min_sensor_dist = 10.0

    def _actionSpace(self):
    # Override the default ±1m destination cap of BaseRLAviary's PID mode
    # treats the action as an absolute world-coordinate destination
        act_lower = np.array([[-2.5, -2.5, 0.0]])
        act_upper = np.array([[ 2.5,  2.5, 2.5]])
        return spaces.Box(low=act_lower, high=act_upper, dtype=np.float32)

    def _computeInfo(self):
        return {} # BaseRLAviary does not have and is mandatory

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

        initial_obs = self._computeObs()

        return initial_obs, info

    def _computeRewardHoverStage(self):
        '''
        Reward policy for a hovering in place drone (not using)
        '''
        pos = self.state[0:3]
        rpy = self.state[7:10]
        vel = self.state[10:13]
        
        if pos[2] < 0.1:
            return -100.0
            
        survival_bonus = 5.0
        altitude_reward = 1.0 - abs(1.0 - pos[2])
        
        velocity_penalty = np.linalg.norm(vel) * 0.5 
        drift_penalty = np.linalg.norm(pos[0:2]) * 0.2
        
        # O antídoto para o suicídio acrobático. 
        # Pune agressivamente a rede se ela tentar inclinar o drone,
        # forçando-a a aprender que a posição mais lucrativa é 100% nivelada.
        acrobatics_penalty = (abs(rpy[0]) + abs(rpy[1])) * 2.0
        
        return float(survival_bonus + altitude_reward - velocity_penalty - drift_penalty - acrobatics_penalty)

    def _computeReward(self):
        pos = self.state[0:3]
        quat = self.state[3:7]
        vel = self.state[10:13] #lin vel

        reward = 0.0

        # --- cost of existing (we want the drone to finish as fast as possible) ---
        reward -= 0.005

        # cost for hiitng the grouund
        # Z < 0.1 
        if pos[2] < 0.1:
            reward -= 40.0

        # cost of collision with pillars
        is_colliding = self._checkPhysicalCollision()
        if is_colliding:
            reward -= 5.0  # Colisão física confirmada (qualquer ângulo, não só frontal)

            
        # cost of being to close to something (caouth by the camera)
        if self.min_sensor_dist < 0.5:
            # Punição exponencial quanto mais perto da parede
            reward -= (0.5 - self.min_sensor_dist) * 1.0 


        # cost of moving in a direction that is not in FOV
        vel_norm = np.linalg.norm(vel)
        if vel_norm > 0.1:
            rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3) # get drone orientation
            forward_vec = rot_matrix.dot(np.array([1.0, 0.0, 0.0])) # 1 0 0 array (front) rotated by orientation (forward dir)
            vel_dir = vel / vel_norm # vx, vy and vz componets (normalized)
            
            # dot product
            # basically we project the 1 0 0 vector in the direction vector and 
            # get the angle between them (alignment)
            # if the angle is greater than the fov, the drone is moving in a direction it cannot see
            alignment = np.dot(forward_vec, vel_dir)
            
            fov_threshold = np.cos(np.deg2rad(self.fov / 2.0))

            if alignment < fov_threshold: 
                reward -= (fov_threshold - alignment) * 1.0

        # reward for getting near the goal
        dist_to_goal = np.linalg.norm(self.goal_pos - pos)
        reward -= dist_to_goal * 0.5

        # reward for getting to the goal
        if dist_to_goal < 0.2:
            reward += 100.0

        
        return float(reward)

    def _checkPhysicalCollision(self):
        """
        Ground-truth collision check via PyBullet's contact manifold.
        Independent of the depth camera's FOV — catches side/rear/below hits
        that min_sensor_dist can never see.
        """
        drone_id = self.DRONE_IDS[0]
        contacts = p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT)

        if len(contacts) == 0:
            return False

        # index 2 of a contact tuple is bodyUniqueIdB
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

        """
        just a getter for the camera, drone state and objective placement
        this is the getter for the NN input
        """

        self.state = self._getDroneStateVector(0)
        safe_state = self.state.copy()
        safe_state[10:13] = np.clip(safe_state[10:13], -10.0, 10.0)
        safe_state[13:16] = np.clip(safe_state[13:16], -20.0, 20.0)

        rel_goal = (self.goal_pos - self.state[0:3]).astype(np.float32)
        kin_obs = np.concatenate([safe_state, rel_goal]).astype(np.float32)

        pos = self.state[0:3]
        quat = self.state[3:7]
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        forward_vec = rot_matrix.dot(np.array([1.0, 0.0, 0.0]))  # 45° down-forward
        up_vec = rot_matrix.dot(np.array([0.0, 0.0, 1.0]))
        target_pos = pos + forward_vec

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vec
        )
        _renderer = p.ER_TINY_RENDERER if self.GUI else p.ER_BULLET_HARDWARE_OPENGL
        _, _, _, depth_img, _ = p.getCameraImage(
            width=self.cam_w, height=self.cam_h,
            viewMatrix=view_matrix, projectionMatrix=self.proj_matrix,
            renderer=_renderer
        )
        depth_array = np.array(depth_img, dtype=np.float32).reshape(self.cam_h, self.cam_w, 1)
        depth_linear = self.far * self.near / (self.far - (self.far - self.near) * depth_array)
        self.min_sensor_dist = np.min(depth_linear)
        image_obs = (depth_array * 255.0).astype(np.uint8).transpose(2, 0, 1)  # → (1, H, W)

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

        # === THE OBSTACLES (Red Pillars) ===
        drone_xy = np.array(self.INIT_XYZS[0][0:2]) if self.INIT_XYZS is not None else np.array([0.0, 0.0])
        goal_xy = np.array(self.goal_pos[0:2])

        min_clearance = 0.5  # pillar radius (0.15) + drone/goal footprint + margin

        for _ in range(4):
            for _attempt in range(20):
                candidate_xy = np.array([
                    np.random.uniform(-1.5, 1.5),
                    np.random.uniform(-1.5, 1.5)
                ])
                clears_drone = np.linalg.norm(candidate_xy - drone_xy) >= min_clearance
                clears_goal = np.linalg.norm(candidate_xy - goal_xy) >= min_clearance
                if clears_drone and clears_goal:
                    break

            pos = [candidate_xy[0], candidate_xy[1], 1.0]

            col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=2.0)
            vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.15, length=2.0, rgbaColor=[0.7, 0.2, 0.2, 1])

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