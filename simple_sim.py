import numpy as np

class MazeSimulation:
    def __init__(self, dt=0.05, arena_size=10.0):
        self.dt = dt
        self.robot_radius = 0.2
        self.arena_size = float(arena_size)
        self.half_arena = self.arena_size / 2.0
        self.arena_diagonal = float(np.sqrt(2.0 * self.arena_size**2))
        
        # 3 Pillars of varied sizes in the compact 10m x 10m arena:
        # - Large Pillar (2.2m x 2.2m) in upper-left quadrant
        # - Small Pillar 1 (1.5m x 1.5m) in lower-right quadrant
        # - Small Pillar 2 (1.2m x 1.2m) in upper-right quadrant
        # Corridors between pillars and outer walls range from 1.8m to 3.1m
        self.pillars = [
            {"name": "large_pillar", "x_min": -3.2, "x_max": -1.0, "y_min": 1.0, "y_max": 3.2},
            {"name": "small_pillar_1", "x_min": 1.2, "x_max": 2.7, "y_min": -3.2, "y_max": -1.7},
            {"name": "small_pillar_2", "x_min": 1.4, "x_max": 2.6, "y_min": 1.4, "y_max": 2.6}
        ]
        
        # Generate wall segments (4 outer boundary walls + 4 walls per pillar = 16 walls total)
        walls_list = [
            # 4 Outer boundary walls
            [-self.half_arena, -self.half_arena,  self.half_arena, -self.half_arena],
            [ self.half_arena, -self.half_arena,  self.half_arena,  self.half_arena],
            [ self.half_arena,  self.half_arena, -self.half_arena,  self.half_arena],
            [-self.half_arena,  self.half_arena, -self.half_arena, -self.half_arena],
        ]
        
        for p in self.pillars:
            walls_list.extend([
                [p["x_min"], p["y_min"], p["x_max"], p["y_min"]],  # Bottom edge
                [p["x_max"], p["y_min"], p["x_max"], p["y_max"]],  # Right edge
                [p["x_max"], p["y_max"], p["x_min"], p["y_max"]],  # Top edge
                [p["x_min"], p["y_max"], p["x_min"], p["y_min"]],  # Left edge
            ])
            
        self.walls = np.array(walls_list, dtype=np.float64)
        
        # Precompute static wall vectors for high-throughput vectorized operations
        self.wall_A = self.walls[:, 0:2]
        self.wall_B = self.walls[:, 2:4]
        self.wall_vec = self.wall_B - self.wall_A
        self.wall_dot = np.sum(self.wall_vec * self.wall_vec, axis=1) + 1e-8
        self.num_walls = len(self.walls)
        
        # Sensor parameters
        # Sensor parameters
        self.num_rays = 15
        self.fov = np.deg2rad(100)
        self.half_fov = self.fov / 2.0
        self.max_range = 10.0
        self.ray_offsets = np.linspace(-self.half_fov, self.half_fov, self.num_rays, dtype=np.float64)
        
        # Physical and velocity limits
        self.max_v = 3.0      # Physical max absolute velocity (m/s)
        self.max_a = 3.0      # Physical max acceleration (m/s^2)
        self.safe_v = 1.5     # Fixed safe velocity threshold (m/s)
        
        # Pose, velocity, target init
        self.pose = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.last_speed = 0.0
        self.impact_speed = 0.0
        self.obj_x = 0.0
        self.obj_y = 0.0
        self.last_min_wall_dist = 0.0
        
        # Randomized initialization ensuring valid non-overlapping spawn
        self.gen_valid_pose()
        self.gen_rand_obj()

    @property
    def speed(self):
        """Current scalar speed in m/s."""
        return float(np.hypot(self.vel[0], self.vel[1]))

    def get_local_velocity(self):
        """Returns velocity in robot local body frame (forward vx, lateral vy)."""
        cos_t = np.cos(-self.pose[2])
        sin_t = np.sin(-self.pose[2])
        vx_local = self.vel[0] * cos_t - self.vel[1] * sin_t
        vy_local = self.vel[0] * sin_t + self.vel[1] * cos_t
        return float(vx_local), float(vy_local)

    def _is_inside_any_pillar(self, x, y, margin=0.0):
        """Checks if point (x, y) falls inside or within margin of any pillar."""
        for p in self.pillars:
            if (p["x_min"] - margin <= x <= p["x_max"] + margin) and \
               (p["y_min"] - margin <= y <= p["y_max"] + margin):
                return True
        return False

    def is_point_valid(self, x, y, min_dist=0.45):
        """Checks if point (x, y) is inside arena bounds and clear of all walls and pillars."""
        if abs(x) > self.half_arena - min_dist or abs(y) > self.half_arena - min_dist:
            return False
        if self._is_inside_any_pillar(x, y, margin=min_dist):
            return False
        if self._get_min_dist_to_walls(x, y) < min_dist:
            return False
        return True

    def gen_valid_pose(self, min_dist=0.45):
        """Spawns drone at a randomized valid location and heading clear of obstacles."""
        while True:
            rx = np.random.uniform(-self.half_arena + min_dist, self.half_arena - min_dist)
            ry = np.random.uniform(-self.half_arena + min_dist, self.half_arena - min_dist)
            if self.is_point_valid(rx, ry, min_dist=min_dist):
                rtheta = np.random.uniform(-np.pi, np.pi)
                self.pose[0] = rx
                self.pose[1] = ry
                self.pose[2] = rtheta
                self.vel[:] = 0.0
                self.last_speed = 0.0
                self.impact_speed = 0.0
                self.last_min_wall_dist = self._get_min_dist_to_walls(rx, ry)
                break

    def gen_rand_obj(self, min_dist=0.45, min_dist_to_robot=1.8):
        """
        Spawns objective at a valid randomized location clear of walls and pillars,
        separated by at least min_dist_to_robot from the drone.
        """
        while True:
            ox = np.random.uniform(-self.half_arena + min_dist, self.half_arena - min_dist)
            oy = np.random.uniform(-self.half_arena + min_dist, self.half_arena - min_dist)
            if self.is_point_valid(ox, oy, min_dist=min_dist):
                dist_to_drone = np.hypot(ox - self.pose[0], oy - self.pose[1])
                if dist_to_drone >= min_dist_to_robot:
                    self.obj_x = ox
                    self.obj_y = oy
                    break

    def step(self, ax, ay, omega):
        """
        Updates 2nd-order kinematics given local body accelerations (ax, ay)
        and yaw rate (omega). Supports active acceleration and deceleration (braking).
        Strictly enforces maximum absolute velocity (self.max_v).
        """
        old_x, old_y, old_theta = self.pose
        cos_t = np.cos(old_theta)
        sin_t = np.sin(old_theta)
        
        # Save previous speed before acceleration update
        self.last_speed = float(np.hypot(self.vel[0], self.vel[1]))
        
        # Convert local agent accelerations to global arena accelerations
        global_ax = ax * cos_t - ay * sin_t
        global_ay = ax * sin_t + ay * cos_t
        
        # Forward Euler velocity update: v = v + a * dt
        self.vel[0] += global_ax * self.dt
        self.vel[1] += global_ay * self.dt
        
        # Clamp to physical maximum absolute velocity
        current_speed = float(np.hypot(self.vel[0], self.vel[1]))
        if current_speed > self.max_v:
            scale = self.max_v / current_speed
            self.vel[0] *= scale
            self.vel[1] *= scale
            current_speed = self.max_v
            
        # Forward Euler position update: p = p + v * dt
        new_x = old_x + self.vel[0] * self.dt
        new_y = old_y + self.vel[1] * self.dt
        new_theta = old_theta + omega * self.dt
        
        # Wrap theta into [-pi, pi]
        new_theta = (new_theta + np.pi) % (2.0 * np.pi) - np.pi
        
        # Check wall collision using vectorized projection and pillar bounds
        dist_parede = self._get_min_dist_to_walls(new_x, new_y)
        inside_pillar = self._is_inside_any_pillar(new_x, new_y, margin=0.0)
        self.last_min_wall_dist = dist_parede
        
        if dist_parede >= self.robot_radius and not inside_pillar:
            self.pose[0] = new_x
            self.pose[1] = new_y
            self.pose[2] = new_theta
            self.impact_speed = 0.0
        else:
            self.pose[2] = new_theta
            # Record impact speed for penalty calculation, then halt linear velocity
            self.impact_speed = current_speed
            self.vel[:] = 0.0

    def is_wp_in_fov(self):
        """Returns True if the target waypoint falls within the sensor FOV cone."""
        rx, ry, rtheta = self.pose
        dx = self.obj_x - rx
        dy = self.obj_y - ry
        cos_t = np.cos(-rtheta)
        sin_t = np.sin(-rtheta)
        local_x = dx * cos_t - dy * sin_t
        local_y = dx * sin_t + dy * cos_t
        angle_to_goal = np.arctan2(local_y, local_x)
        return bool(abs(angle_to_goal) <= self.half_fov)

    def _get_min_dist_to_walls(self, x, y):
        """Fast vectorized point-to-segment distance to all walls."""
        AP_x = x - self.wall_A[:, 0]
        AP_y = y - self.wall_A[:, 1]
        dot_AP_AB = AP_x * self.wall_vec[:, 0] + AP_y * self.wall_vec[:, 1]
        t = np.clip(dot_AP_AB / self.wall_dot, 0.0, 1.0)
        closest_x = self.wall_A[:, 0] + t * self.wall_vec[:, 0]
        closest_y = self.wall_A[:, 1] + t * self.wall_vec[:, 1]
        dx = x - closest_x
        dy = y - closest_y
        return float(np.sqrt(np.min(dx * dx + dy * dy)))

    def read_sensors(self):
        """Vectorized LiDAR raycaster calculating all 15 rays against 16 walls simultaneously."""
        pose_x, pose_y, pose_theta = self.pose
        angles = pose_theta + self.ray_offsets
        ray_dirs_x = np.cos(angles)
        ray_dirs_y = np.sin(angles)
        
        # 2D cross products (determinants) shape (num_rays, num_walls)
        det = ray_dirs_x[:, None] * self.wall_vec[None, :, 1] - ray_dirs_y[:, None] * self.wall_vec[None, :, 0]
        
        diff_x = self.wall_A[:, 0] - pose_x
        diff_y = self.wall_A[:, 1] - pose_y
        
        num_t = diff_x * self.wall_vec[:, 1] - diff_y * self.wall_vec[:, 0]
        num_u = diff_x[None, :] * ray_dirs_y[:, None] - diff_y[None, :] * ray_dirs_x[:, None]
        
        valid_det = np.abs(det) > 1e-6
        safe_det = np.where(valid_det, det, 1.0)
        
        t = np.where(valid_det, num_t[None, :] / safe_det, np.inf)
        u = np.where(valid_det, num_u / safe_det, -1.0)
        
        valid_hits = (t > 0.0) & (u >= 0.0) & (u <= 1.0)
        t_valid = np.where(valid_hits, t, self.max_range)
        distances = np.clip(np.min(t_valid, axis=1), 0.0, self.max_range)
        return distances, angles

if __name__ == "__main__":
    from Render import Renderer
    sim = MazeSimulation(arena_size=10.0)
    ui = Renderer(sim)
    ui.run()