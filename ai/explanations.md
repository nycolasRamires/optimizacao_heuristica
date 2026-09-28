# Technical Explanations & Rationale (`ai/explanations.md`)

This document records in-depth technical analyses, mathematical formulations, and engineering rationales behind technical tasks executed in the project.

---

## 1. High-Throughput Simulation & SAC Training Optimization

### Objective & Target Constraints
To enable automated reward tuning via a **Genetic Algorithm (GA)**, each candidate evaluation requires a fast training cycle. The user target required a **1,000,000-timestep training session to complete in under 30 minutes** on an Intel Core Ultra 9 285K and NVIDIA GeForce RTX 5070.

### Baseline Performance Assessment
Before optimization, profiling a baseline 10,000-step training sample yielded:
- **Environment Simulation Speed:** 8,708 FPS
- **SAC Training Speed:** 166.9 steps/s
- **Projected 1M Steps Duration:** **99.88 minutes (~1 hour 40 minutes)** $\to$ *Exceeded budget by 3.3x.*

### Bottleneck Breakdown
Profiling isolated two primary bottlenecks:
1. **Simulation Geometry & Sensor Raycasting:**
   - In [`simple_sim.py`](file:///home/nycolas/m/simple_sim.py), `read_sensors()` evaluated 15 LiDAR rays against 5 walls via nested Python loops ($15 \times 5 = 75$ iterations/step) computing scalar 2D cross products (determinants).
   - In `is_wp_in_fov()`, solving a $2 \times 2$ linear system using `np.linalg.solve(sight, obj)` executed an LAPACK routine every step, creating substantial overhead.
   - In [`rl_training.py`](file:///home/nycolas/m/rl_training.py), wall distance `_get_min_dist_to_walls()` was executed **twice** per step (once in physics update, once in reward computation).
   - Trigonometric transformations (`dx, dy, cos, sin, hypot, arctan2`) for goal coordinates were calculated in `step()` and re-evaluated identically in `_get_observation()`.
2. **RL Training Framework Overhead (PyTorch / SAC):**
   - With a single environment (`env = OmniDroneEnv()`), SB3's default configuration executes 1 gradient update for every 1 environment step.
   - For batch size 256, sampling from CPU memory and executing PyTorch CUDA kernels (actor, 2 critics, target network Polyak update) creates CPU-GPU synchronization latency of ~5.8 ms per step, leaving the RTX 5070 at $<5\%$ utilization.
   - CPU execution (`device="cpu"`) suffered from OpenMP thread contention across 24 cores on tiny batch operations.

---

### Technical Rationale & Mathematical Formulations

#### 1. Vectorized LiDAR Raycasting (`read_sensors`)
For a ray starting at $\mathbf{p}_0 = [x, y]^T$ with heading angles $\alpha_i$ ($i \in \{0, \dots, 14\}$):
$$\mathbf{d}_i = \begin{bmatrix} \cos(\alpha_i) \\ \sin(\alpha_i) \end{bmatrix}$$
And a wall segment from $\mathbf{A}_j$ to $\mathbf{B}_j$ ($j \in \{0, \dots, 4\}$) with vector $\mathbf{w}_j = \mathbf{B}_j - \mathbf{A}_j$:
The intersection satisfies:
$$\mathbf{p}_0 + t_{i,j} \mathbf{d}_i = \mathbf{A}_j + u_{i,j} \mathbf{w}_j$$
Solving by 2D determinants:
$$\text{det}_{i,j} = d_{i,x} w_{j,y} - d_{i,y} w_{j,x}$$
$$t_{i,j} = \frac{(\mathbf{A}_j - \mathbf{p}_0)_x w_{j,y} - (\mathbf{A}_j - \mathbf{p}_0)_y w_{j,x}}{\text{det}_{i,j}}$$
$$u_{i,j} = \frac{(\mathbf{A}_j - \mathbf{p}_0)_x d_{i,y} - (\mathbf{A}_j - \mathbf{p}_0)_y d_{i,x}}{\text{det}_{i,j}}$$

Instead of nested loops, this is expressed as a single $(15, 5)$ NumPy matrix operation:
- Precomputed static wall arrays: $\mathbf{A} \in \mathbb{R}^{5 \times 2}$, $\mathbf{B} \in \mathbb{R}^{5 \times 2}$, $\mathbf{w} \in \mathbb{R}^{5 \times 2}$, and $\|\mathbf{w}\|^2 \in \mathbb{R}^5$.
- Determinant matrix: $\mathbf{D} = \mathbf{d}_x[:, \text{None}] \cdot \mathbf{w}_y[\text{None}, :] - \mathbf{d}_y[:, \text{None}] \cdot \mathbf{w}_x[\text{None}, :]$.
- Intersection masks: Valid when $|\text{det}| > 10^{-6}$, $t > 0$, and $0 \le u \le 1$.
- Range reduction: $r_i = \min_j(\text{where}(\text{valid}, t_{i,j}, \text{max\_range}))$.
- **Verification:** Max absolute difference between scalar and vectorized raycaster is $< 6.3 \times 10^{-15}$ (machine float precision).

#### 2. Analytical FOV Validation (`is_wp_in_fov`)
Given the relative heading $\theta_{\text{goal}} = \text{arctan2}(y_{\text{local}}, x_{\text{local}})$ already computed in the robot's local frame:
$$\text{in\_fov} \iff |\theta_{\text{goal}}| \le \frac{\text{FOV}}{2}$$
This avoids all matrix allocation and LAPACK solver calls, yielding a 7x speedup with 100% exact boolean identity.

#### 3. Cached Wall Distance Projections
During `MazeSimulation.step()`, point-to-segment distance to all walls is already computed to prevent wall clipping. Saving `self.last_min_wall_dist` allows `OmniDroneEnv` to directly check collision and proximity rewards without calling `_get_min_dist_to_walls()` a second time.

#### 4. Vectorized Environment Execution (`DummyVecEnv`)
To eliminate GPU synchronization starvation, the environment is parallelized using `DummyVecEnv([make_env for _ in range(N)])`:
- Steps $N$ environments sequentially in-process, avoiding Python `multiprocessing` IPC/pickling overhead.
- Because the analytical environment steps at $>40{,}000$ FPS, stepping 16 envs in memory is orders of magnitude faster than inter-process pipe communication.
- Batches experience collection, saturates GPU pipeline, and reduces gradient step frequency amortized over transitions.

---

### Empirical Optimization Results

#### Micro-Benchmarks (Simulation Subroutines)
| Subroutine | Baseline (calls/s) | Optimized (calls/s) | Speedup |
| :--- | :--- | :--- | :--- |
| `read_sensors()` (LiDAR) | 16,503 | **79,094** | **+379% (4.8x)** |
| `is_wp_in_fov()` (Target cone) | 144,915 | **1,021,652** | **+605% (7.0x)** |
| `_get_min_dist_to_walls()` | 119,105 | **185,795** | **+56% (1.6x)** |
| `OmniDroneEnv.step()` (Full step) | 8,708 FPS | **40,751 FPS** | **+368% (4.7x)** |

#### Macro-Benchmarks (SAC End-to-End Training)
| Configuration | Throughput (steps/s) | Time for 1,000,000 Steps | Status |
| :--- | :--- | :--- | :--- |
| Baseline (1 env, default train.py) | 166.9 | **99.88 min** | ❌ Fails target |
| Optimized Sim (1 env) | 172.2 | **96.78 min** | ❌ Fails target |
| Optimized Sim (4 envs DummyVecEnv) | 682.0 | **24.44 min** | ✅ Meets target |
| Optimized Sim (8 envs DummyVecEnv) | 1,331.2 | **12.52 min** | ✅ Meets target |
| Optimized Sim (12 envs DummyVecEnv) | 1,954.8 | **8.53 min** | ✅ Meets target |
| **Optimized Sim (16 envs DummyVecEnv)** | **2,584.8** | **6.45 min** | 🚀 **4.6x faster than target** |

---

## 2. Base Pre-Training & Warm-Started Genetic Algorithm Optimization

### Motivation: Cutting Corners in Evolutionary Search
In standard Genetic Algorithm (GA) approaches to Reinforcement Learning reward tuning, training each candidate individual from a randomly initialized network for 500,000 to 1,000,000 steps creates a prohibitive computational burden. For a modest population of 20 individuals over 5 generations (100 total training runs), 500k steps per run would require 50 million steps ($\approx 5.5$ hours even at 2,500 steps/s).

To bypass this bottleneck:
1. A single **`base` policy** is pre-trained for 500,000 timesteps using default balanced reward parameters.
2. This `base` network internalizes basic sensorimotor grounding: interpreting 15 LiDAR ranges, orienting toward targets, and maintaining continuous forward thrust.
3. Every subsequent GA candidate is warm-started from `base.zip` and fine-tuned for a much shorter horizon (e.g. 50,000–100,000 steps).

### The Dynamics of Learning Rate Adaptation During Fine-Tuning
When warm-starting from a pre-trained network:
- The actor and critic networks already possess strong feature representations in their hidden layers.
- If fine-tuned with a default conservative learning rate ($\eta = 3 \times 10^{-4}$), gradient updates are small, and the actor's behavior remains dominated by the inertia of the pre-trained Q-function. The subtle differences between candidate reward weights are washed out or take too long to manifest.
- By **elevating the learning rate** (e.g. $\eta = 1 \times 10^{-3}$), the policy rapidly updates its action distribution and critic landscape to reflect the candidate's unique penalty-to-reward ratios (such as aggressive obstacle repulsion vs. direct velocity optimization).

### PyTorch / Stable-Baselines3 Checkpoint Loading Quirk
Stable-Baselines3 serializes the PyTorch optimizer state dictionaries inside model zip archives. Calling:
```python
model = SAC.load("base.zip", env=vec_env, learning_rate=1e-3)
```
updates the high-level `model.lr_schedule` callable, but does **not** modify the underlying `param_groups['lr']` of the loaded PyTorch optimizers (`actor.optimizer`, `critic.optimizer`, `ent_coef_optimizer`). 

To ensure the new learning rate immediately takes effect in the neural network gradients, [`set_learning_rate`](file:///home/nycolas/m/train.py#L23-L37) iterates through all optimizer parameter groups and explicitly overrides their `lr` values:
```python
for param_group in model.actor.optimizer.param_groups:
    param_group["lr"] = new_lr
for param_group in model.critic.optimizer.param_groups:
    param_group["lr"] = new_lr
if model.ent_coef_optimizer is not None:
    for param_group in model.ent_coef_optimizer.param_groups:
        param_group["lr"] = new_lr
```

---

## 3. High-Difficulty 10m x 10m Arena Geometry, 3 Asymmetric Pillars, & Safe Spawning

### Geometry Specifications & Evolution
To prevent policy complacency and create an environment requiring active LiDAR steering and obstacle dodging, the arena was refined to a compact **$10\text{m} \times 10\text{m}$ map** ($100\text{ m}^2$, coordinates $[-5.0, 5.0] \times [-5.0, 5.0]$) featuring **three asymmetric pillars**:
- **Pillar 1 (Large):** $2.2\text{m} \times 2.2\text{m}$, centered in upper-left ($[-3.2, -1.0] \times [1.0, 3.2]$).
- **Pillar 2 (Small 1):** $1.5\text{m} \times 1.5\text{m}$, centered in lower-right ($[1.2, 2.7] \times [-3.2, -1.7]$).
- **Pillar 3 (Small 2):** $1.2\text{m} \times 1.2\text{m}$, centered in upper-right ($[1.4, 2.6] \times [1.4, 2.6]$).
- **Total Walls:** 4 outer boundaries $+ 3 \times 4$ segments per pillar $= 16$ line segments.
- **Corridor Clearances:**
  - Gap between Large and Small 2: **$2.40\text{m}$**
  - Gap between Small 2 and Small 1: **$3.10\text{m}$**
  - Left boundary clearance: **$1.80\text{m}$**
  - Top boundary clearance: **$1.80\text{m}$**
  - Bottom boundary clearance: **$1.80\text{m}$**
  - Right boundary clearances: **$2.30\text{m}$–$2.40\text{m}$**
- **Obstacle Density:** The 3 pillars occupy $\approx 37.5\%$ of the map area. With $10.0\text{m}$ LiDAR range covering the entire diagonal ($\approx 14.14\text{m}$), the agent continuously perceives surrounding obstacles.

### Interior Pillar Rejection Problem & Formulation
When obstacles are represented as closed polygonal perimeters (collections of 1D wall segments), using point-to-line segment distance alone creates a dangerous failure mode:
$$\text{dist}(\mathbf{p}, \partial P_k) = \min_{\mathbf{s} \in \partial P_k} \|\mathbf{p} - \mathbf{s}\|$$
If a coordinate $\mathbf{p} = [x, y]^T$ falls inside a $2.2\text{m} \times 2.2\text{m}$ pillar, its distance to the 4 exterior boundary segments can be up to $1.1\text{m}$. A naive check ($\text{dist} > 0.45\text{m}$) would falsely classify the pillar interior as valid open space, causing the drone or target to spawn embedded inside solid concrete.

To guarantee safe spawning, rejection sampling enforces a strict dual condition:
1. **Minkowski Expanded Box Rejection:**
   $$\forall k \in \{1, 2, 3\}, \quad \mathbf{p} \notin [x_{\min}^k - \delta, x_{\max}^k + \delta] \times [y_{\min}^k - \delta, y_{\max}^k + \delta]$$
   where $\delta = 0.45\text{m}$ provides safety clearance from the pillar faces.
2. **Boundary Wall Distance:**
   $$\min_{j} \text{dist}(\mathbf{p}, \text{wall}_j) \ge \delta$$
3. **Inter-Entity Minimum Distance:**
   $$\|\mathbf{p}_{\text{objective}} - \mathbf{p}_{\text{drone}}\| \ge 1.8\text{m}$$

### Topological Accessibility
Because none of the 3 pillars connect to the outer boundary walls or to each other, the navigable workspace:
$$\mathcal{W}_{\text{free}} = [-5, 5]^2 \setminus (P_1 \cup P_2 \cup P_3)$$
is a single, path-connected 2D manifold with three holes. Every corridor exceeds $1.8\text{m}$ in width—more than $4.5\times$ the drone diameter ($0.4\text{m}$). For any valid spawn pose $\mathbf{p}_{\text{start}} \in \mathcal{W}_{\text{free}}$ and target $\mathbf{p}_{\text{goal}} \in \mathcal{W}_{\text{free}}$, there exists an infinite family of collision-free continuous paths $\gamma: [0, 1] \to \mathcal{W}_{\text{free}}$ connecting them. The reward is mathematically guaranteed to always be accessible.

---

## 4. Baseline Policy Pre-Training Horizon (300k Steps) & Benchmark Evaluation

### 300k Training Execution Metrics
- **Environment:** 16 parallel environments in `DummyVecEnv`, $10\text{m} \times 10\text{m}$ arena with 3 asymmetric pillars.
- **Hardware:** Intel Core Ultra 9 285K (24 cores) + NVIDIA GeForce RTX 5070 (PyTorch CUDA).
- **Execution Time:** **2 minutes 02 seconds** (122 seconds total).
- **Throughput:** **2,455 iterations/second** average.
- **Artifact:** [`base.zip`](file:///home/nycolas/m/base.zip).

### Deterministic Benchmark (100 Seeded Evaluation Episodes)
To characterize the baseline ancestor policy prior to GA genetic optimization, the checkpoint was evaluated under deterministic policy inference ($\mu(s)$):

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Success Rate** | **64.0%** (64/100) | Successfully maneuvers around pillars and reaches target within 0.25m. |
| **Collision Rate** | **29.0%** (29/100) | Clips pillar corners or outer arena boundaries during sharp turns. |
| **Timeout Rate** | **7.0%** (7/100) | Stagnates in corridors or local dead ends exceeding 500 steps. |
| **Avg Steps to Goal** | **159.8 steps** | Efficient navigation when obstacle clearance is maintained. |

### Rationale for Downscaling to 300k
1. **Preserving Policy Plasticity:** At 500k+ timesteps, SAC's actor and critic networks approach local convergence where the policy distribution sharpens (entropy coefficient $\alpha \to 0$). At 300k steps, the entropy coefficient ($\alpha \approx 0.0159$) maintains sufficient stochasticity and gradient sensitivity to allow rapid adaptation when fine-tuned under novel GA reward profiles.
2. **Room for Evolutionary Differentiation:** With a 64% success baseline, individual candidate reward genomes have clear room to demonstrate improvement (e.g., reducing the 29% collision rate through higher wall proximity penalties, or reducing the 7% timeout rate through stronger progress incentives). If the baseline were already 95%+ optimal, evaluating fitness differences among GA candidates would require thousands of evaluation episodes to detect statistically significant variances.

---

## 5. Existing Reward & Penalty Policy Formulation

The environment reward $R_t$ at timestep $t$ in [`OmniDroneEnv`](file:///home/nycolas/m/rl_training.py) is a linear combination of continuous shaping terms and event-triggered terminal signals:

$$R_t = R_{\text{progress}} + R_{\text{fov}} + R_{\text{goal}} + R_{\text{early}} - P_{\text{living}} - P_{\text{prox}} - P_{\text{stag}} - P_{\text{loiter}} - P_{\text{retreat}} - P_{\text{coll}}$$

### Parameter Breakdown & Mathematical Definitions

| Parameter Key | Default Value | Mathematical Term / Condition | Behavioral Role & Rationale |
| :--- | :--- | :--- | :--- |
| **`living_cost`** | `0.01` | $P_{\text{living}} = c_{\text{living}}$ (every step) | Constant step penalty discouraging time waste, idle hovering, or circuitous paths. Forces the agent to find time-efficient trajectories. |
| **`goal_reward`** | `100.0` | $R_{\text{goal}} = r_{\text{goal}} \cdot \mathbb{I}(\|\mathbf{p} - \mathbf{p}_{\text{target}}\| < 0.25)$ | Primary sparse objective signal. Triggers episode success termination (`terminated = True`, `is_success = True`). Large positive magnitude anchors the return scale. |
| **`early_goal_multiplier`** | `0.1` | $R_{\text{early}} = \max(0, T_{\max} - t) \cdot k_{\text{early}}$ (at goal) | Timestep-based bonus rewarding early arrival: fewer timesteps spent yield more points upon reaching the goal, while strictly maintaining the fixed `goal_reward`. |
| **`collision_penalty`** | `1.0` | $P_{\text{coll}} = c_{\text{coll}} \cdot \mathbb{I}(d_{\text{wall}} \le r_{\text{robot}} + 0.01 \lor \text{inside})$ | Catastrophic termination penalty for striking walls or pillar boundaries. Triggers episode failure (`terminated = True`, `collision = True`). |
| **`wall_proximity_penalty`** | `0.05` | $P_{\text{prox}} = c_{\text{prox}} \cdot \mathbb{I}(d_{\text{wall}} \le r_{\text{robot}} + 0.075)$ | Soft repulsive buffer penalty within a $7.5\text{cm}$ envelope around obstacles. Encourages the drone to maintain a safety margin before an actual collision occurs. |
| **`progress_multiplier`** | `10.0` | $R_{\text{progress}} = (d_{t-1} - d_t) \cdot k_{\text{prog}} \cdot \max(f_{\min}, \cos \theta_{\text{local}})$ | Dense potential-based shaping term. Proportional to the Euclidean distance reduction toward the goal, modulated by heading alignment. Moving closer yields positive reward; moving away yields negative penalty. |
| **`direction_factor_min`** | `0.1` | $f_{\min} = 0.1$, modulates heading weighting | Sets the floor for progress weighting. Even if the omnidirectional drone travels toward the goal sideways or backward ($\cos \theta_{\text{local}} \le 0$), it retains at least $10\%$ progress incentive, but receives maximum progress reward when its front LiDAR points directly at the target ($\cos 0 = 1$). |
| **`retreat_tolerance`** | `0.05` | $\delta_{\text{tol}} = 0.05\text{m}$ ($5\text{cm}$) | Slack margin for allowable distance increase relative to recent minimum distance. Accommodates necessary lateral circumventing maneuvers around pillar boundaries. |
| **`retreat_penalty`** | `0.02` | Flat component of retreat penalty | Base penalty applied when distance increase relative to recent steps exceeds $\delta_{\text{tol}}$. |
| **`retreat_multiplier`** | `2.0` | $k_{\text{retreat}} \cdot (d_t - d_{\min, W} - \delta_{\text{tol}})$ | Proportional penalty scaling with the excess distance drifted away from the goal beyond tolerance. |
| **`retreat_window_steps`** | `10` | Rolling history window $W = \{d_{t-1}, \dots, d_{t-K}\}$ | Number of recent steps over which minimum goal distance is tracked ($K=10 \approx 0.167\text{s}$). |
| **`fov_bonus`** | `0.01` | $R_{\text{fov}} = r_{\text{fov}} \cdot \mathbb{I}(|\theta_{\text{local}}| \le \text{half\_fov})$ | Bonus for keeping the target inside the $100^\circ$ forward LiDAR visual cone ($|\theta_{\text{local}}| \le 50^\circ$). Encourages intentional forward-facing navigation and sensor alignment. |
| **`stagnation_penalty`** | `0.01` | $P_{\text{stag}} = c_{\text{stag}} \cdot \mathbb{I}(|d_{t-1} - d_t| < \frac{r_{\text{robot}}}{15})$ | Applied when the change in distance to goal is negligible ($< 1.33\text{cm}$). Discourages deadlocking or oscillating against walls without making headway. |
| **`loitering_penalty`** | `0.1` | $P_{\text{loiter}} = c_{\text{loiter}} \cdot \mathbb{I}(|v_x| + |v_y| + |\omega| < 0.1)$ | Control penalty when all commanded velocities drop close to zero. Prevents policies from freezing in safe zones to escape living and collision penalties. |

### Mechanics of the New Policies

#### 1. Retreat Penalty Relative to Recent Steps with Tolerance
Navigating around convex obstacles (pillars) geometrically requires an agent to temporarily increase its Euclidean distance to the goal to clear an obstacle corner:
$$d_t > d_{t-1}$$
If any instantaneous distance increase is strictly punished, policies can get trapped against pillar faces. To resolve this:
1. Maintain a rolling ring buffer of recent goal distances: $W = [d_{t-K}, \dots, d_{t-1}]$, where $K = \text{retreat\_window\_steps}$ (default 10 steps, $\approx 0.167\text{s}$).
2. Determine the local minimum distance: $d_{\min} = \min_{d \in W} d$.
3. Compute distance increase: $\Delta d_{\text{retreat}} = d_t - d_{\min}$.
4. If $\Delta d_{\text{retreat}} > \delta_{\text{tol}}$:
   $$\text{excess} = \Delta d_{\text{retreat}} - \delta_{\text{tol}}$$
   $$P_{\text{retreat}} = c_{\text{retreat}} + k_{\text{retreat}} \cdot \text{excess}$$
This allows local maneuvers within $5\text{cm}$ of slack, but penalizes substantial or sustained retreat.

#### 2. Early Arrival Bonus with Fixed Terminal Reward
To directly incentivize rapid navigation without disrupting the value scale of the sparse success signal:
$$R_{\text{terminal}} = R_{\text{goal}} + \max(0, T_{\max} - t) \cdot k_{\text{early}}$$
- If the drone reaches the goal at $t = 100$ ($T_{\max} = 500$, $k_{\text{early}} = 0.1$):
  $$R_{\text{terminal}} = 100.0 + (500 - 100) \times 0.1 = 140.0$$
- If the drone reaches the goal at $t = 480$:
  $$R_{\text{terminal}} = 100.0 + (500 - 480) \times 0.1 = 102.0$$
- The fixed $R_{\text{goal}} = 100.0$ is strictly preserved in both cases.

---

## 6. 2nd-Order Acceleration Kinematics & Velocity Safety Policies

### Dynamic State & Forward Euler Integration
Under 2nd-order dynamics, the drone possesses both position $\mathbf{p} = [x, y]^T$, orientation $\theta$, and translational velocity $\mathbf{V} = [V_x, V_y]^T \in \mathbb{R}^2$. The control input commanded by the neural network consists of local longitudinal and lateral accelerations plus angular rate:
$$\mathbf{u}_t = [a_x, a_y, \omega]^T \in [-1, 1]^3$$
Scaled to physical units:
$$a_x = u_0 \cdot a_{\max}, \quad a_y = u_1 \cdot a_{\max}, \quad \omega = u_2 \cdot \omega_{\max}$$
where $a_{\max} = 3.0\text{ m/s}^2$ and $\omega_{\max} = 3.0\text{ rad/s}$.

1. **Rotation to Global Frame:**
   $$\begin{bmatrix} A_x \\ A_y \end{bmatrix} = \begin{bmatrix} \cos \theta_t & -\sin \theta_t \\ \sin \theta_t & \cos \theta_t \end{bmatrix} \begin{bmatrix} a_x \\ a_y \end{bmatrix}$$
2. **Velocity Update & Hard Physical Saturation:**
   $$\tilde{\mathbf{V}}_{t+1} = \mathbf{V}_t + \mathbf{A}_t \Delta t$$
   $$\mathbf{V}_{t+1} = \begin{cases} \tilde{\mathbf{V}}_{t+1} & \text{if } \|\tilde{\mathbf{V}}_{t+1}\| \le v_{\max} \\ \frac{v_{\max}}{\|\tilde{\mathbf{V}}_{t+1}\|} \tilde{\mathbf{V}}_{t+1} & \text{if } \|\tilde{\mathbf{V}}_{t+1}\| > v_{\max} \end{cases}$$
   where $v_{\max} = 2.0\text{ m/s}$. The drone is physically constrained from exceeding $2.0\text{ m/s}$ under any action.
3. **Position & Heading Update:**
   $$\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{V}_{t+1} \Delta t, \quad \theta_{t+1} = \text{wrap}(\theta_t + \omega_t \Delta t)$$
4. **Collision Inelastic Stopping:**
   Upon wall or pillar boundary impact ($d \le r_{\text{robot}}$), the pre-collision impact speed $v_{\text{impact}} = \|\mathbf{V}_{t+1}\|$ is recorded for penalty computation, and velocity is inelastically clamped to $\mathbf{V}_{t+1} = \mathbf{0}$.

### 19D Observation Space Expansion
To maintain the Markov property ($\mathbb{P}(s_{t+1} | s_t, a_t) = \mathbb{P}(s_{t+1} | s_0, \dots, s_t, a_t)$), the observation vector was expanded from 17D to 19D:
$$\mathbf{s}_t = [s_0, \dots, s_{14}, s_{15}, s_{16}, s_{17}, s_{18}]^T \in \mathbb{R}^{19}$$
- **$s_0 \dots s_{14}$:** 15 normalized LiDAR range measurements $\in [0, 1]$.
- **$s_{15}$:** Normalized Euclidean distance to target $\in [0, 1]$.
- **$s_{16}$:** Normalized target bearing angle in robot frame $\in [-1, 1]$.
- **$s_{17}$:** Normalized local longitudinal velocity $v_{x,\text{local}} / v_{\max} \in [-1, 1]$.
- **$s_{18}$:** Normalized local lateral velocity $v_{y,\text{local}} / v_{\max} \in [-1, 1]$.

### Speed-Dependent Policies Formulation

| Policy Term | Key & Default | Mathematical Expression | Behavioral Objective |
| :--- | :--- | :--- | :--- |
| **High Speed Reward** | `high_speed_reward: 0.02` | $R_{\text{speed}} = k_{\text{spd}} \cdot \frac{v_t}{v_{\max}}$ | Continuous incentive for flying fast; offsets living cost when moving briskly. |
| **Crash Speed Penalty** | `crash_speed_penalty: 2.0` | $P_{\text{crash}} = k_{\text{crash}} \cdot \frac{v_{\text{impact}}}{v_{\max}}$ | Additional impact penalty proportional to momentum at wall strike. |
| **High Speed Crash Penalty** | `high_speed_crash_penalty: 5.0` | $P_{\text{hi\_crash}} = k_{\text{hi\_crash}} \cdot \frac{v_{\text{impact}} - v_{\text{safe}}}{v_{\max} - v_{\text{safe}}}$ (if $v > v_{\text{safe}}$) | Severe quadratic-like spike in penalty when crashing above fixed safety threshold $v_{\text{safe}} = 1.2\text{m/s}$. |
| **Overspeed Flight Penalty** | `overspeed_penalty: 0.05` | $P_{\text{over}} = k_{\text{over}} \cdot \frac{v_t - v_{\text{safe}}}{v_{\max} - v_{\text{safe}}}$ (if $v > v_{\text{safe}}$) | Per-step penalty applied during free flight whenever speed exceeds fixed $v_{\text{safe}} = 1.2\text{m/s}$. |
| **Constant Velocity Reward** | `constant_vel_reward: 0.02` | $R_{\text{const}} = k_{\text{const}} \cdot \max\left(0, 1 - \frac{\|v_t - v_{t-1}\|}{a_{\max} \Delta t}\right)$ | Rewards steady cruising when $v_t > 0.2\text{m/s}$, penalizing erratic throttle oscillation and jerky maneuvers. |

### Acceleration Baseline Benchmark (300k Timesteps)
- **Training Time:** 2 minutes 04 seconds (2,408 steps/s) across 16 parallel environments in `DummyVecEnv`.
- **Checkpoint:** [`base.zip`](file:///home/nycolas/m/base.zip)

#### Quantitative Evaluation (100 Seeded Episodes)
| Metric | Value | Behavioral Analysis |
| :--- | :--- | :--- |
| **Success Rate** | **39.0%** (39/100) | Directly navigates through multi-pillar corridors to reach the goal under inertial dynamics. |
| **Collision Rate** | **25.0%** (25/100) | Fails to decelerate in time when approaching corners or slips into pillar edges. |
| **Timeout Rate** | **36.0%** (36/100) | Policy decelerates too cautiously or enters local oscillatory loops in tight corridors. |
| **Avg Steps to Goal** | **258.7 steps** | Longer paths due to inertial turning radii and deceleration before target capture. |
| **Avg Flight Speed** | **0.91 m/s** | Successfully stabilizes below the fixed $v_{\text{safe}} = 1.2\text{ m/s}$ threshold to avoid overspeed penalties. |

This provides an ideal genetic starting point: the drone has mastered the basic physics of acceleration, braking, and speed regulation, but the 36% timeout and 25% collision rates offer strong gradients for the Genetic Algorithm to optimize reward hyperparameters.

---

## 7. Genetic Algorithm Reward Optimization Architecture (`evolution.py`)

### Optimization Formulation
The Genetic Algorithm (GA) optimizes the 19-dimensional continuous reward parameter vector:
$$\boldsymbol{\theta} \in \Theta \subset \mathbb{R}^{19}$$
such that a warm-started SAC policy fine-tuned for $T_{\text{fine}} = 500,000$ steps maximizes deterministic navigation fitness:
$$\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta} \in \Theta} \mathcal{F}(\pi_{\boldsymbol{\theta}})$$
where $\pi_{\boldsymbol{\theta}}$ is initialized from [`base.zip`](file:///home/nycolas/m/base.zip) (300k steps) and fine-tuned under reward profile $\boldsymbol{\theta}$ with elevated learning rate ($\alpha_{\text{fine}} = 10^{-3}$) to reach 800k total experience steps.

### Concurrency & Hardware Execution Model
To maximize throughput on the 24-core Intel Core Ultra 9 285K and NVIDIA GeForce RTX 5070:
- **Concurrency:** 2 individuals are trained concurrently in parallel worker processes via `multiprocessing.get_context("spawn")`.
- **Worker Environment Allocation:** Each worker runs an 8-environment `DummyVecEnv` (16 vectorized environments active simultaneously).
- **GPU Resource Footprint:** SAC actor and critic networks require $\approx 1.2\text{ GB}$ of VRAM per worker ($<2.5\text{ GB}$ total out of 12 GB available), avoiding GPU memory bottlenecks.
- **Estimated Epoch Duration:** A batch of 2 individuals ($2 \times 500\text{k}$ steps) finishes in $\approx 3.5$ minutes. A generation of 10 individuals finishes in $\approx 17.5$ minutes. 20 generations require $\approx 5.8$ hours.

### Genome Representation & Search Bounds (`GENE_BOUNDS`)
The 19 genes cover all active shaping and penalty terms:

| Gene / Parameter | Bounds $[low, high]$ | Dtype | Search Purpose & Rationale for Expansion |
| :--- | :--- | :--- | :--- |
| `living_cost` | $[0.001, 0.075]$ | float | Calibrates time pressure vs. safe cautiousness. Expanded up to $0.075$ (+50%). |
| `collision_penalty` | $[0.1, 15.0]$ | float | Base wall/pillar contact penalty. Expanded down to $0.1$ and up to $15.0$. |
| `crash_speed_penalty` | $[0.1, 15.0]$ | float | Momentum-dependent crash penalty scale. Expanded up to $15.0$. |
| `high_speed_crash_penalty`| $[1.0, 15.0]$ | float | Penalty spike for crashing above $v_{\text{safe}}$. |
| `wall_proximity_penalty` | $[0.001, 0.40]$ | float | Soft repulsive barrier strength. Doubled upper bound from $0.20 \to 0.40$ to avoid border saturation. |
| `progress_multiplier` | $[1.0, 35.0]$ | float | Gradient steepness toward objective. Expanded up to $35.0$ (+40%) to accommodate aggressive guidance. |
| `stagnation_penalty` | $[0.005, 0.10]$ | float | Anti-deadlock penalty. Doubled upper bound from $0.05 \to 0.10$. |
| `loitering_penalty` | $[0.01, 0.75]$ | float | Anti-freezing control penalty. Expanded from $0.50 \to 0.75$. |
| `high_speed_reward` | $[0.001, 0.20]$ | float | Speed promotion in clear corridors. Doubled upper bound to $0.20$. |
| `overspeed_penalty` | $[0.01, 0.30]$ | float | Safety enforcement above $v_{\text{safe}} = 1.2\text{m/s}$. Expanded from $0.20 \to 0.30$. |
| `constant_vel_reward` | $[0.001, 0.15]$ | float | Steady cruising smoothness incentive. Expanded from $0.10 \to 0.15$. |
| `goal_reward` | $[25.0, 250.0]$ | float | Primary task completion anchor. Broadened from $[50, 200]$ to $[25, 250]$. |
| `early_goal_multiplier` | $[0.01, 0.75]$ | float | Timestep arrival bonus. Expanded from $0.50 \to 0.75$. |
| `fov_bonus` | $[0.001, 0.10]$ | float | Sensor orientation guidance. Expanded from $0.07 \to 0.10$. |
| `direction_factor_min` | $[0.01, 1.0]$ | float | Lateral vs. forward progress weighting floor. Expanded up to $1.0$ (allows strict forward heading). |
| `retreat_penalty` | $[-0.01, 0.01]$ | float | Flat retreat penalty/bonus component. Doubled range from $\pm 0.005 \to \pm 0.010$. |
| `retreat_multiplier` | $[0.5, 10.0]$ | float | Scaled retreat excess penalty. |


*Note: `retreat_tolerance` ($0.05\text{m}$) and `retreat_window_steps` ($10\text{ steps} \approx 0.167\text{s}$) are held constant across all individuals as fixed physical simulation parameters.*

### Objective Fitness Function & Viability Survival Gate
Rather than evaluating a composite scalar that depends on internal reward scaling, evaluation directly measures physical task efficiency across $N = 100$ deterministic seeded evaluation tries:

1. **Viability Threshold Gate:**
   $$\text{Viable}(\pi) = \begin{cases} \text{True} & \text{if } \text{SR} \ge 0.50 \\ \text{False} & \text{if } \text{SR} < 0.50 \end{cases}$$
   - Any individual achieving $\text{SR} < 50\%$ dies immediately.
   - It is disqualified from ranking, elitism, and reproduction.
   - Its slot in the population is replenished by a newly sampled random individual (random immigrant).

2. **Physical Navigation Fitness Formulation:**
   For all surviving individuals ($\text{SR} \ge 0.50$), fitness is defined as the average timesteps required to reach the target across successful episodes:
   $$\mathcal{F}(\pi) = \bar{T}_{\text{goal}} = \frac{1}{|\mathcal{S}_{\text{success}}|} \sum_{i \in \mathcal{S}_{\text{success}}} T_i$$
   where $T_i \in [1, 500]$ is the completion step count for episode $i$.
   - **Ranking Rule:** Strictly sorted in ascending order of $\mathcal{F}$ (fewer timesteps = higher rank / faster navigation).
   - Because $|\mathcal{S}_{\text{success}}| \ge 50$ for every surviving individual, $\bar{T}_{\text{goal}}$ is statistically robust against outlier initializations.

### Multi-Group Evolutionary Reproduction Architecture

The population transition from generation $g$ to $g+1$ for $N = 20$ individuals implements the user's multi-group reproduction policy:

```
[Generation g (N = 20)]
         │
         ├───> Success Rate < 0.50 ───> n_dead individuals (Disqualified & Die)
         │
         └───> Success Rate >= 0.50 ──> n_survivors (Ranked by avg timesteps to goal: Rank 1 is fastest)
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┬──────────────────────────────┐
                 ▼                              ▼                              ▼                              ▼
             Group 2:                       Group 3:                       Group 4:                       Group 1:
        Top n_best = 4                 Elite Combinations             Rank Selection                 Recombination
         Direct Clones                 C(4, 2) = 6 pairs             (Non-elite pool)               (Survivor pool)
         (Unmutated)                  12 sons -> 6 chosen             Without Repl.                 n_dead offspring
                                       (Mutated, 3-blend)             (Unmutated)                   (Mutated, 3-blend)
```

#### 1. Group Allocation & Size Formulas

Given $N = 20$, $n_{\text{best}} = 4$, $\binom{n_{\text{best}}}{2} = 6$:

- **Group 2 (Elites Passing Directly):**
  $$k_2 = \min(n_{\text{best}}, n_{\text{survivors}}) = 4$$
  The top 4 survivors pass directly to generation $g+1$ as unmutated clones, preserving optimal evaluated reward parameters intact.
- **Group 3 (Elite Recombination Offspring):**
  $$k_3 = \binom{k_2}{2} = 6$$
  All $\binom{4}{2} = 6$ distinct unordered pairs of elites are recombined. Each pair produces 2 children ($2 \times 6 = 12$ total). Exactly half ($6$) are randomly sampled to pass to generation $g+1$ after Gaussian mutation.
- **Group 4 (Rank-Selected Non-Elite Survivors):**
  $$k_4 = \max\left(0, N - k_2 - k_3 - n_{\text{dead}}\right)$$
  When $n_{\text{dead}} < 10$, $k_4 > 0$. Candidates are drawn without replacement from the non-elite survivors ($\text{Rank } 5 \dots n_{\text{survivors}}$).
  The selection probability for candidate $i$ at 1-indexed survivor rank $r_i$ is weighted by:
  $$W_{r_i} = n_{\text{survivors}} - r_i + 1, \quad P(i) = \frac{W_{r_i}}{\sum_{j \in \text{pool}} W_{r_j}}$$
  Selected individuals pass directly as unmutated clones.
- **Group 1 (Recombination Offspring Replacing Dead):**
  $$k_1 = N - (k_2 + k_3 + k_4)$$
  Offspring are generated by sampling random pairs from all survivors with replacement. Each pair produces 2 children. Offspring undergo 3-gene blend crossover and Gaussian mutation to replenish the population back to exactly $N = 20$.

#### 2. High-Mortality Fallback Guarantee ($n_{\text{dead}} > 10$)
If $> 10$ individuals fail the $50\%$ viability threshold:
- Available survivors are prioritized: top surviving elites ($k_2 \le 4$) and elite offspring ($k_3 = \binom{k_2}{2} \le 6$) are preserved.
- Group 4 receives $k_4 = 0$ slots.
- All remaining $N - k_2 - k_3$ slots are filled by random survivor recombination ($k_1$).
- If $n_{\text{survivors}} = 1$: the single elite passes and remaining 19 slots are re-seeded with fresh random genomes.
- If $n_{\text{survivors}} = 0$: all 20 slots are re-seeded with fresh random genomes.

#### 3. 3-Gene Blend Crossover Operator (`recombine_parents`)
For parents $P_1, P_2 \in \mathbb{R}^{17}$:
1. A random subset of $n_{\text{blend}} = 3$ distinct gene keys $\mathcal{B} \subset \{g_1, \dots, g_{17}\}$ is sampled without replacement.
2. For each gene $g \in \mathcal{B}$:
   $$\alpha \sim \mathcal{U}(0, 1)$$
   $$C_1[g] = \text{clamp}\left(\alpha P_1[g] + (1 - \alpha) P_2[g], \text{low}_g, \text{high}_g\right)$$
   $$C_2[g] = \text{clamp}\left((1 - \alpha) P_1[g] + \alpha P_2[g], \text{low}_g, \text{high}_g\right)$$
3. For the remaining $17 - 3 = 14$ genes ($g \notin \mathcal{B}$), uniform discrete crossover is applied:
   $$(C_1[g], C_2[g]) = \begin{cases} (P_1[g], P_2[g]) & \text{with probability } 0.5 \\ (P_2[g], P_1[g]) & \text{with probability } 0.5 \end{cases}$$

#### 4. Bounded Gaussian Mutation Operator (`mutate_genes`)
For each gene $g$ in an individual undergoing mutation:
With probability $p_{\text{mut}} = 0.10$ (10% fixed rate):
$$\Delta_g \sim \mathcal{N}\left(0, \left(0.1 \cdot (\text{high}_g - \text{low}_g)\right)^2\right)$$
$$x_{\text{mut}}[g] = \text{clamp}\left(x[g] + \Delta_g, \text{low}_g, \text{high}_g\right)$$
If the gene has integer dtype, the mutated value is rounded to the nearest integer. Elites (Group 2) and rank-selected survivors (Group 4) pass unmutated.

---

## 8. Empirical Analysis of the 5-Generation Pilot Run & Convergence Dynamics

### Run Overview
- **Setup:** 5 generations, $N = 20$ individuals/generation, 300,000 fine-tune timesteps/individual starting from [`base.zip`](file:///home/nycolas/m/base.zip), evaluated over 100 test episodes.
- **Hardware Execution:** Wall-clock time per generation averaged $\approx 39.8\text{ minutes}$ across 2 parallel workers. Total run duration was $3.3\text{ hours}$.

### Performance Summary
| Generation | Best Fitness (Steps) | Population Mean Steps | Best Success Rate | Mean Success Rate | Best Collision Rate | Mean Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **147.1** (`Ind 06`) | $171.9 \pm 18.4$ | 63.0% | 75.2% | 32.0% | 1.69 m/s |
| **1** | **131.0** (`Ind 03`) ★ | $173.2 \pm 20.6$ | 66.0% | 73.8% | 24.0% | 1.74 m/s |
| **2** | **139.9** (`Ind 10`) | $179.0 \pm 15.8$ | 63.0% | 72.8% | 31.0% | 1.67 m/s |
| **3** | **153.6** (`Ind 11`) | $174.3 \pm 15.6$ | 66.0% | 75.5% | 30.0% | 1.73 m/s |
| **4** | **141.0** (`Ind 03`) | $173.8 \pm 15.5$ | 61.0% | 73.3% | 30.0% | 1.70 m/s |

### Convergence Diagnosis: Why the Population Did Not Settle in 5 Generations
1. **Dimensionality vs. Sample Budget:**
   A 17-dimensional continuous optimization space cannot converge within 100 total policy evaluations ($5 \times 20$). In genetic algorithms, early generations (generations 0–5) correspond to the *broad exploration* phase, while convergence typically occurs across generations 12–25 as recombination focuses around high-performing attractors.
2. **Zero Mortality & Maximum Diversity Retention:**
   The $50\%$ success rate viability threshold proved easily achievable for fine-tuned policies (all individuals scored $54\% - 89\%$). With $0$ dead individuals in generations 1–4, Group 4 took all 10 remaining slots to preserve non-elite survivors across the ranking spectrum. This maintained strong diversity, preventing premature convergence at the expense of early variance.
3. **Emergent Selection Gradients:**
   Clear directional drift was already visible in several key genes:
   - `progress_multiplier`: Steadily rose from $12.04 \to 15.95$ (+32%), demonstrating selection for steep target gradients.
   - `loitering_penalty`: Increased from $0.212 \to 0.327$ (+54%), penalizing drone hesitation.
   - `wall_proximity_penalty`: Rose from $0.117 \to 0.145$ (+24%), forcing cautious corner clearance.

---

## 9. Algorithmic Upgrades: Bounded Relative Mutation, Tournament Selection, and 17-Gene Evolution Dynamics

### 1. Bounded $\pm 50\%$ Relative Mutation Operator
Previously, Gaussian mutation added noise proportional to the whole search domain range:
$$\Delta_g \sim \mathcal{N}(0, (0.1 \cdot (\text{high}_g - \text{low}_g))^2)$$
For wide domains (e.g., `goal_reward` in $[50, 200]$), this caused abrupt, destructive shifts that broke finely-tuned policies. The upgraded mutation operator applies a controlled relative shift:
$$x_{\text{mut}}[g] = \text{clamp}\left(x[g] + s \cdot \Delta_r, \text{low}_g, \text{high}_g\right)$$
where:
- Direction $s \in \{-1, +1\}$ is chosen uniformly at random ($50\%$ chance up, $50\%$ chance down).
- Relative perturbation magnitude:
  $$\Delta_r = \begin{cases} \beta \cdot |x[g]| & \text{if } |x[g]| > 10^{-4} \\ \beta \cdot 0.1 \cdot (\text{high}_g - \text{low}_g) & \text{if near zero} \end{cases}$$
  with $\beta \sim \mathcal{U}(0, 0.50)$ (up to a $50\%$ increase or decrease).
This ensures localized exploration near high-performing parameter regions while respecting hard domain bounds.

### 2. Tournament Selection for Non-Elite Survivors ($k = 3$)
To heighten selective pressure while preventing premature collapse, Group 4 replaces rank-proportional selection with deterministic tournament selection without replacement:
1. Candidate pool $\mathcal{S}$ initialized with all non-elite survivors.
2. For each slot $i \in \{1, \dots, n_{\text{slots}}\}$:
   - Randomly draw $k = 3$ competitors from $\mathcal{S}$ uniformly without replacement.
   - Winner $w = \arg\min_{c \in \text{tournament}} \text{timesteps\_to\_goal}(c)$.
   - Add $w$ to Group 4 and remove $w$ from $\mathcal{S}$ to guarantee unique survivors.
Tournament selection with $k = 3$ provides a steeper selection gradient than linear rank weighting, accelerating the propagation of superior non-elite traits.

### 3. Increased Viability Gate (55% Success Rate)
The survival threshold was raised from $0.50 \to 0.55$:
$$\text{is\_viable} = (\text{success\_rate} \ge 0.55)$$
Over 100 evaluation episodes, individuals achieving $\le 54\%$ success die immediately. In the 5-generation dataset, this gate successfully eliminated Generation 0 Individual 07 ($\text{SR} = 47.0\%$).

### 4. 17-Gene Allele Drift and Empirical Selection Analysis
Across the 5-generation run, tracking all 17 genes revealed clear behavioral selection pressures:
- **`loitering_penalty` ($+44.8\%$):** $0.226 \to 0.327$ — drones that hesitate at pillar bottlenecks were systematically out-competed.
- **`early_goal_multiplier` ($+41.1\%$):** $0.278 \to 0.392$ — rewarded policies reaching the target significantly before timeout.
- **`progress_multiplier` ($+36.6\%$):** $11.7 \to 15.9$ — high potential gradient driving aggressive traversal along corridor paths.
- **`wall_proximity_penalty` ($+28.6\%$):** $0.113 \to 0.145$ — selected for maintaining a critical safety cushion around pillars.
- **`direction_factor_min` ($-23.9\%$):** $0.271 \to 0.206$ — relaxed heading restrictions, unlocking agile lateral and diagonal omnidirectional sliding maneuvers.

---

## 10. High-Throughput Optimization Dynamics: Numba JIT, SAC Pipelining, and CUDA Compilation

### 1. The Per-Step SAC Update Bottleneck & `train_freq` Pipelining
In default Stable-Baselines3 SAC (`train_freq=1`, `gradient_steps=1`), execution switches between the Python CPU environment rollout and the PyTorch CUDA optimizer every single timestep. Because launching a CUDA kernel has a fixed CPU dispatch overhead ($\approx 10-20\,\mu\text{s}$ per kernel), running 1 gradient step per environment step forces constant CPU-GPU context switches and stalls:
$$\text{Rollout Step} \to \text{Sample Replay Buffer} \to \text{Critic Update} \to \text{Actor Update} \to \text{Temp Update} \to \text{Rollout Step}$$
By configuring:
$$\text{train\_freq} = (4, \text{"step"}), \quad \text{gradient\_steps} = 4$$
- The **Update-to-Data Ratio (UTD)** is strictly preserved: $\text{UTD} = \frac{4}{4} = 1.0$. The neural network receives the exact same number of optimization steps per collected experience transition, preserving sample efficiency.
- PyTorch can pipeline 4 consecutive mini-batch forward/backward passes on CUDA without CPU-environment interruptions, eliminating GPU pipeline bubbling.

### 2. Native Compilation via Numba (`@njit(fastmath=True)`)
Although the 2D kinematic engine in [`simple_sim.py`](file:///home/nycolas/m/simple_sim.py) previously used vectorized NumPy matrix operations, invoking NumPy array allocations and broadcasting inside tight loops still incurred CPython interpreter overhead.
- `_numba_read_sensors`: Directly loops across 15 rays $\times$ 16 walls in native machine code.
  - Latency dropped from $155.6\,\text{ms} \to 5.04\,\text{ms}$ per 10,000 calls (**$30.8\times$ speedup**).
- `_numba_min_dist_to_walls`: Unrolled point-to-segment projection across all 16 walls with early squared-distance pruning.
  - Latency dropped from $56.7\,\text{ms} \to 1.86\,\text{ms}$ per 10,000 calls (**$30.5\times$ speedup**).

### 3. Policy Actor Compilation (`torch.compile`) & Serialization Safety
- Compiling `model.policy.actor` with PyTorch Inductor generates optimized CUDA kernels for the 19D $\to$ 256 $\to$ 256 $\to$ 6 MLP forward pass, lowering rollout inference latency.
- **Serialization Safety:** PyTorch Inductor wraps compiled modules in `OptimizedModule`, creating an internal `_orig_mod` attribute. Standard SB3 `model.save()` would serialize keys with the `_orig_mod` prefix, breaking uncompiled loading. To resolve this, `train.py` and `evolution.py` automatically unwrap `model.policy.actor = model.policy.actor._orig_mod` prior to `model.save()`, guaranteeing $100\%$ portable `.zip` checkpoints.

### 4. Empirical Throughput Benchmark Results
Benchmarked on RTX 5070 across 20,000 training timesteps:
- **Baseline (Pre-optimization, 16 envs):** $2,450.2\text{ steps/s}$ ($8.16\text{s}$)
- **Post-optimization (16 envs):** $3,025.6\text{ steps/s}$ ($6.61\text{s}$, **$+23.5\%$ speedup**)
- **Post-optimization (20 envs):** $3,801.5\text{ steps/s}$ ($5.26\text{s}$, **$+55.1\%$ speedup**)
At $3,800\text{ steps/s}$, fine-tuning a 300,000-step individual takes only **$\approx 79\text{ seconds}$**. For a 30-individual generation with 2 parallel workers, active training completes in **$\approx 20\text{ minutes}$ per generation**.

---

## 11. Extended Evolutionary Optimization Run (30 Generations x 25 Individuals)

### Run Overview & Scaled Configuration
- **Total Generations:** 30 (Generations 00 through 29).
- **Population Size:** 25 individuals/generation (750 total trained candidate policies).
- **Fine-Tuning Budget:** 300,000 steps per candidate ($225\text{M}$ simulation steps total across the run) warm-started from [`base.zip`](file:///home/nycolas/m/base.zip).
- **Hardware & Throughput:** Dual-worker multiprocessing (`concurrency=2`) with 20 vectorized environments per worker (`envs_per_worker=20`, 40 environments total) on Intel Core Ultra 9 285K and NVIDIA GeForce RTX 5070. Average generation duration was $\approx 23.6\text{ minutes}$.
- **Evolutionary Configuration:**
  - $n_{\text{best}} = 5$ elite clones passed unmutated.
  - $C(5, 2) = 10$ elite recombination offspring ($20$ children generated, half selected, 3-gene blend crossover).
  - Group 4: Tournament selection ($k=3$) for non-elite survivors.
  - Group 1: Random survivor recombination replacing dead individuals.
  - Bounded $\pm 50\%$ relative mutation ($p_{\text{mut}} = 0.02$).
  - Viability threshold: $\text{SR} \ge 55\%$.

### Key Findings & Dual Champions

#### 1. All-Time Evolutionary Champion (Across All 30 Generations)
- **Generation:** Gen 09 | **Individual ID:** 12 (`elite_recomb`)
- **Fitness (Avg Steps to Goal):** **128.7 steps**
- **Evaluation Metrics (100 episodes):**
  - Success Rate: **66.0%**
  - Collision Rate: **34.0%**
  - Timeout Rate: **0.0%**
  - Average Flight Speed: **1.86 m/s**
- **Model Checkpoint:** [`ga_results/gen_09/ind_12/model.zip`](file:///home/nycolas/m/ga_results/gen_09/ind_12/model.zip)
- **Evolved Genome Highlights:**
  - `early_goal_multiplier`: **0.4990** (near upper bound $0.50$, heavily driving time-minimizing trajectory convergence).
  - `progress_multiplier`: **19.97** (strong gradient incentive toward objective).
  - `collision_penalty`: **7.67** and `high_speed_crash_penalty`: **8.18** (high collision aversion offsetting speed incentives).
  - `living_cost`: **0.0259** (moderate pace pressure).

#### 2. Best Individual of Final Generation (Generation 29)
- **Generation:** Gen 29 | **Individual ID:** 07 (`elite_recomb`)
- **Fitness (Avg Steps to Goal):** **141.5 steps**
- **Evaluation Metrics (100 episodes):**
  - Success Rate: **70.0%**
  - Collision Rate: **26.0%**
  - Timeout Rate: **4.0%**
  - Average Flight Speed: **1.92 m/s**
- **Model Checkpoint:** [`ga_results/gen_29/ind_07/model.zip`](file:///home/nycolas/m/ga_results/gen_29/ind_07/model.zip)
- **Behavioral Shift:** Gen 29 Ind 07 trades off a small amount of raw traversal speed ($141.5$ vs $128.7$ steps) in exchange for higher safety clearance, reducing collision rate from $34.0\% \to 26.0\%$ and increasing success rate to $70.0\%$.

### Population Dynamics & Survival Statistics
- **Mortality Rate:** Consistently low across all 30 generations ($0$ to $3$ dead per generation, averaging $<8\%$ mortality). The population maintained $>90\%$ survival above the $55\%$ viability threshold.
- **Mean Generation Steps:** Stabilized in the $160$–$175$ steps range, while elite lineages consistently achieved $135$–$145$ steps.
- **Visual Analytics:** The complete genealogy, fitness dispersion, population composition, and 17-gene convergence grid are packaged in the standalone interactive report [`ga_results/evolution_dashboard.html`](file:///home/nycolas/m/ga_results/evolution_dashboard.html).

---

## 12. Deep Diagnosis: Convergence, Generational Fluctuation, Mutation Scope, and High-SR Trade-offs

### 1. Gene Convergence vs. Non-Convergence
Tracking the population allele spread from Generation 00 to Generation 29 reveals that **8 out of 17 genes achieved strong convergence** (variance reduction $>75\%$ to $99.9\%$):
- **`high_speed_crash_penalty`:** $\sigma$ dropped from $3.82 \to 0.003$ (**$99.9\%$ reduction**), settling tightly at $\approx 1.84$.
- **`retreat_penalty`:** $\sigma$ dropped from $0.0031 \to 0.0000$ (**$99.9\%$ reduction**), locking at $-0.0012$.
- **`constant_vel_reward`:** $\sigma$ dropped from $0.0286 \to 0.0019$ (**$93.4\%$ reduction**), converging at $\approx 0.0087$.
- **`early_goal_multiplier`:** Population bulk converged at $\approx 0.1387$ ($\sigma$ dropped from $0.130 \to 0.0058$, **$95.5\%$ reduction**).
- **`loitering_penalty`:** $\sigma$ dropped from $0.132 \to 0.0137$ (**$89.6\%$ reduction**), converging at $\approx 0.0886$.
- **`stagnation_penalty`:** $\sigma$ dropped from $0.0120 \to 0.0018$ (**$85.0\%$ reduction**), converging at $\approx 0.0176$.
- **`retreat_multiplier`:** $\sigma$ dropped from $2.91 \to 0.56$ (**$80.8\%$ reduction**), converging at $\approx 3.32$.
- **`fov_bonus`:** $\sigma$ dropped from $0.0167 \to 0.0038$ (**$77.2\%$ reduction**), converging at $\approx 0.030$.

**Unconverged / Broad Variance Genes:**
- `goal_reward` ($\sigma = 24.1$ in Gen 29): Dense shaping terms (`progress_multiplier` $\approx 20.5$ and `early_goal_multiplier`) dominate credit assignment, leaving the raw constant terminal bonus with relatively flat selective gradient across $[50, 150]$.
- `direction_factor_min` ($0.35$–$0.50$): Policies can succeed both with strict front alignment and with flexible lateral omnidirectional crabbing.

### 2. Why Does Plot 1A Show High Inter-Generational Variance?
Despite gene convergence, individual ranking positions oscillate between generations due to two factors:
1. **Stochastic RL Re-Training Noise:**
   When an elite clone passes to generation $g+1$, its neural network weights are *not* preserved; it is re-fine-tuned from `base.zip` for 300,000 steps under its candidate reward function. Because SAC involves stochastic exploration, replay buffer batch sampling, and environment seeds, training the exact same genome twice produces an inherent empirical variance of $\pm 25$–$50$ evaluation steps.
   - *Example:* Gen 03 Rank 1 (Ind 05, $136.2$ steps, $60\%$ SR) produced an identical clone in Gen 04 (Ind 00) that scored $186.6$ steps ($56\%$ SR), tumbling from Rank 1 down to Rank 20 solely due to RL training stochasticity.
2. **Evaluation Sample Size Noise:**
   Even over 100 evaluation episodes, randomized spawn and target configurations across the 3-pillar arena introduce a standard error of $\approx \pm 5$–$10$ steps.

### 3. Mutation Mechanism Analysis: Gene-Level vs. Individual-Level
- **Current Implementation:** In `mutate_genes()`, every recombined offspring (Group 3 and Group 1) is evaluated across all 17 genes in a loop: `if np.random.rand() < mutation_rate: mutate(gene)`.
- **Root Cause of Excessive ⚡ Badges:**
  There is **no individual-level mutation gate**. With 17 genes, the probability of at least one gene mutating is:
  $$P(\ge 1 \text{ mutation}) = 1 - (1 - p_{\text{gene}})^{17}$$
  Even at $p_{\text{gene}} = 0.02$, $29.1\%$ of all recombined children mutate at least one gene. At $p_{\text{gene}} = 0.05$, $58.2\%$ mutate; at $p_{\text{gene}} = 0.10$, $83.3\%$ mutate.
  Plot 1A flags any individual where `len(mutated_genes) > 0` with a `⚡` badge, making mutation appear ubiquitous.
- **Architectural Solution:** Implement a hierarchical two-tier mutation operator:
  1. *Individual Gate:* Draw $u \sim \mathcal{U}(0, 1)$. Only if $u < P_{\text{ind}}$ (e.g. $0.20$) does the individual mutate.
  2. *Gene Mutation:* For selected individuals, mutate $1$ randomly selected gene (or mutate each gene with $p = 1/17 \approx 0.059$).

### 4. Resolving the "Fast vs. High Success Rate" Dilemma
- **Root Cause of Reckless Champions:**
  The fitness function was formulated as:
  $$\text{Fitness} = \bar{T}_{\text{goal}} \quad \text{for all } \pi \text{ with } \text{SR} \ge 0.55$$
  Because the ranking sorted strictly by $\bar{T}_{\text{goal}}$, success rate carried **zero weight** once an individual cleared 55%. A reckless policy reaching the goal in 128 steps with 34% crashes was ranked superior to a robust policy reaching the goal in 142 steps with only 17% crashes.
- **Discovered High-SR + Fast Policies in the Database:**
  Filtering the 750 trained policies for both high SR ($\ge 80\%$) and fast speed yields superior candidates:
  - **[`Gen 23 Ind 04`](file:///home/nycolas/m/ga_results/gen_23/ind_04/model.zip):** **83.0% SR**, 17.0% Collisions, **142.3 steps**, **1.94 m/s**.
  - **[`Gen 29 Ind 10`](file:///home/nycolas/m/ga_results/gen_29/ind_10/model.zip):** **82.0% SR**, 18.0% Collisions, **142.4 steps**, **1.94 m/s**.
  - **[`Gen 06 Ind 00`](file:///home/nycolas/m/ga_results/gen_06/ind_00/model.zip):** **87.0% SR**, 11.0% Collisions, **147.3 steps**, **2.00 m/s**.
  - **[`Gen 27 Ind 09`](file:///home/nycolas/m/ga_results/gen_27/ind_09/model.zip):** **78.0% SR**, 22.0% Collisions, **136.0 steps**, **1.95 m/s**.
- **Genome Differentiator:**
  The reckless 66% SR champion had `early_goal_multiplier = 0.4990` (near upper bound $0.50$), creating suicidal time pressure. The 82%–83% SR candidates calibrated `early_goal_multiplier` around $0.097$–$0.130$ and maintained a maximum `wall_proximity_penalty` ($0.1974$).
- **Recommended Fitness Formulation for Next GA Runs:**
  **Penalized Effective Timesteps:** Assign maximum episode steps ($T_{\max} = 500$) to collisions/timeouts:
  $$\mathcal{F}_{\text{effective}}(\pi) = \frac{1}{N} \sum_{i=1}^N \begin{cases} T_i & \text{if success} \\ 500 & \text{if collision or timeout} \end{cases}$$
  Under this metric:
  - Gen 09 Ind 12 ($66\%$ SR, $128.7$ steps): $0.66 \times 128.7 + 0.34 \times 500 = \mathbf{254.9\text{ effective steps}}$.
  - Gen 23 Ind 04 ($83\%$ SR, $142.3$ steps): $0.83 \times 142.3 + 0.17 \times 500 = \mathbf{203.1\text{ effective steps}}$ (**Clear Winner!**).
  - Gen 06 Ind 00 ($87\%$ SR, $147.3$ steps): $0.87 \times 147.3 + 0.13 \times 500 = \mathbf{193.2\text{ effective steps}}$ (**All-Time Best!**).

---

## 13. Mathematical Formulations: Hierarchical Mutation, Dynamic Schedules, and Penalized Effective Timesteps

### 1. Hierarchical Two-Tier Mutation Operator
To eliminate the artifact where almost every recombined offspring received a mutation badge in Plot 1A, the mutation operator is decoupled into an **individual-level gate** and a **targeted gene-level perturbation**:

1. **Individual Selection Gate:**
   For each recombined offspring $C$ (in Group 3 and Group 1):
   $$u \sim \mathcal{U}(0, 1)$$
   $$C \text{ is selected for mutation} \iff u < P_{\text{mut}}(g)$$
   If $u \ge P_{\text{mut}}(g)$, the child passes unmutated ($C$ inherits exact blended crossover chromosomes with $0$ mutation flags).

2. **Gene Selection (50% 1-Gene / 50% 2-Genes):**
   If $C$ is selected for mutation:
   $$K \sim \begin{cases} 1 & \text{with probability } 0.50 \\ 2 & \text{with probability } 0.50 \end{cases}$$
   Exactly $K$ gene keys $\mathcal{M} = \{g_1, \dots, g_K\} \subset \text{GENES}$ are sampled uniformly at random without replacement ($K \in \{1, 2\}$ out of $17$).

3. **Dynamic Annealed Gene Perturbation:**
   For each selected gene $g \in \mathcal{M}$:
   $$s \sim \{-1, +1\} \quad \text{with equal probability } 0.50$$
   $$\delta_g = \begin{cases} s \cdot B(g) \cdot |x[g]| & \text{if } |x[g]| > 10^{-6} \\ s \cdot B(g) \cdot 0.10 \cdot (\text{high}_g - \text{low}_g) & \text{otherwise} \end{cases}$$
   $$x_{\text{mut}}[g] = \text{clamp}\left(x[g] + \delta_g, \text{low}_g, \text{high}_g\right)$$
   (Rounded to nearest integer if gene dtype is integer).

### 2. Generational Dynamic Schedules Formulation
Let $g \in \{0, 1, \dots, G - 1\}$ denote the 0-indexed generation number across $G$ total generations, with normalized progress:
$$\tau(g) = \frac{g}{\max(1, G - 1)} \in [0, 1]$$

| Parameter | Symbol | Initial Value ($\tau=0$) | Final Value ($\tau=1$) | Mathematical Formulation | Evolutionary Objective |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Individual Mutation Rate** | $P_{\text{mut}}(g)$ | $0.10$ (10%) | $0.20$ (20%) | $P_{\text{mut}}(g) = 0.10 + 0.10 \cdot \tau(g)$ | Promotes recombination exploitation early; increases exploration late to counteract diversity loss. |
| **Mutation Perturbation Band** | $B(g)$ | $1.00$ ($\pm 100\%$) | $0.25$ ($\pm 25\%$) | $B(g) = 1.00 - 0.75 \cdot \tau(g)$ | Broad parameter space exploration early; fine-grained localized tuning near convergence. |
| **Viability Survival Gate** | $V(g)$ | $0.50$ (50% SR) | $0.65$ (65% SR) | $V(g) = 0.50 + 0.15 \cdot \tau(g)$ | Lenient filter during early locomotion shaping; strict survivability standards in late generations. |

### 3. Penalized Effective Timesteps Fitness Formulation
To solve the conflict where reckless policies (e.g. 66% SR, 128 steps) beat highly reliable navigation policies (e.g. 87% SR, 147 steps), the fitness function treats non-successes (collisions and timeouts) identically as reaching the episode horizon limit ($T_{\max} = 500$ steps):

For each evaluation episode $i \in \{1, \dots, N\}$ ($N = 100$):
$$T_i^{\text{effective}} = \begin{cases} T_i & \text{if episode succeeded (target captured within } 0.25\text{m)} \\ T_{\max} = 500 & \text{if collision with wall/pillar or timeout occurred} \end{cases}$$

$$\mathcal{F}_{\text{effective}}(\pi) = \frac{1}{N} \sum_{i=1}^N T_i^{\text{effective}}$$

**Mathematical Ranking Rule:**
Survivors ($\text{SR} \ge V(g)$) are sorted strictly in ascending order of $\mathcal{F}_{\text{effective}}(\pi)$:
$$\pi_1 \succ \pi_2 \iff \mathcal{F}_{\text{effective}}(\pi_1) < \mathcal{F}_{\text{effective}}(\pi_2)$$

**Comparative Effect on Policy Selection:**
- **Reckless Policy ($66\%$ SR, $\bar{T} = 128.7\text{ steps}$):**
  $$\mathcal{F}_{\text{eff}} = 0.66 \times 128.7 + 0.34 \times 500 = 84.9 + 170.0 = \mathbf{254.9\text{ steps}}$$
- **Balanced Policy ($83\%$ SR, $\bar{T} = 142.3\text{ steps}$):**
  $$\mathcal{F}_{\text{eff}} = 0.83 \times 142.3 + 0.17 \times 500 = 118.1 + 85.0 = \mathbf{203.1\text{ steps}} \quad (\mathbf{51.8\text{ steps superior!}})$$
- **High-Reliability Policy ($87\%$ SR, $\bar{T} = 147.3\text{ steps}$):**
  $$\mathcal{F}_{\text{eff}} = 0.87 \times 147.3 + 0.13 \times 500 = 128.2 + 65.0 = \mathbf{193.2\text{ steps}} \quad (\mathbf{61.7\text{ steps superior!}})$$

Under this metric, the evolutionary engine directly selects policies that minimize flight time while penalizing crashes with maximal severity, driving the population toward the ideal Pareto frontier of speed and reliability.

---

## 14. Expanded Gene Search Space Bounds: Dimensional Diversity & Holonomic Dynamics

### 1. Motivation for Bound Expansion
In the initial 30-generation run, evolutionary trajectories for several critical parameters exhibited boundary saturation:
- **`progress_multiplier`:** Consistently drifted toward the upper boundary of $25.0$, suggesting that the drone policy's forward motivation remained constrained.
- **`direction_factor_min`:** Was capped at $0.50$, forcing the reward function to heavily favor forward-facing motion ($r_{\text{progress}} \propto \max(0.5, \cos\theta_{\text{goal}})$). While appropriate for non-holonomic vehicles (cars, standard planes), an omnidirectional drone has 3-DOF translation independent of yaw, meaning forcing yaw alignment imposes unnecessary rotational delays.
- **`collision_penalty` & `crash_speed_penalty`:** Capped at $10.0$, which proved insufficient to deter high-speed wall impacts when balanced against aggressive progress gains.
- **`retreat_penalty`:** Bounded strictly positive ($[0.005, 0.10]$), which penalized any temporary backward drift even when needed to maneuver around tight pillar corners.

### 2. Comprehensive Comparison Table of Search Space Bounds

| Gene Key | Previous Bounds | Expanded Bounds | Dynamic Range Multiplier | Behavioral & Rationale Impact |
| :--- | :--- | :--- | :--- | :--- |
| `living_cost` | $[0.001, 0.05]$ | $[0.001, 0.075]$ | $1.5\times$ | Increases pressure to avoid hovering or lingering in empty space. |
| `collision_penalty` | $[0.5, 10.0]$ | $[0.1, 15.0]$ | $1.57\times$ | Allows exploration from mild collision tolerance to extreme zero-tolerance crash aversion. |
| `crash_speed_penalty` | $[0.5, 10.0]$ | $[0.1, 15.0]$ | $1.57\times$ | Provides wide dynamic range for penalizing reckless high-velocity impacts. |
| `high_speed_crash_penalty` | $[1.0, 15.0]$ | $[1.0, 15.0]$ | $1.0\times$ | Retained stable; already provides substantial penalty for impacts above $v_{\text{safe}} = 1.2\text{ m/s}$. |
| `wall_proximity_penalty` | $[0.01, 0.20]$ | $[0.001, 0.40]$ | $2.1\times$ | Allows discovery of either tight corridor cutters or ultra-conservative wide-clearance fliers. |
| `progress_multiplier` | $[2.0, 25.0]$ | $[1.0, 35.0]$ | $1.48\times$ | Unleashes strong forward pursuit gradient to accelerate flight speed across long straightaways. |
| `stagnation_penalty` | $[0.005, 0.05]$ | $[0.005, 0.10]$ | $2.11\times$ | Prevents stationary oscillation or loitering near pillar corners. |
| `loitering_penalty` | $[0.02, 0.50]$ | $[0.01, 0.75]$ | $1.54\times$ | Stronger deterrent against drone freezing when confronting narrow corridors. |
| `high_speed_reward` | $[0.005, 0.10]$ | $[0.001, 0.20]$ | $2.09\times$ | Encourages sprint velocities approaching the physical max speed ($v_{\max} = 2.0\text{ m/s}$). |
| `overspeed_penalty` | $[0.01, 0.20]$ | $[0.01, 0.30]$ | $1.53\times$ | Provides counter-balance against uncontrolled acceleration beyond safe limits. |
| `constant_vel_reward` | $[0.005, 0.10]$ | $[0.001, 0.15]$ | $1.57\times$ | Stabilizes smooth cruise flight, reducing jerky acceleration oscillations. |
| `goal_reward` | $[50.0, 200.0]$ | $[25.0, 250.0]$ | $1.5\times$ | Expands the relative dominance of goal capture vs intermediate waypoint shaping. |
| `early_goal_multiplier` | $[0.02, 0.50]$ | $[0.01, 0.75]$ | $1.54\times$ | Directly incentivizes sprinting to the waypoint in fewer timesteps. |
| `fov_bonus` | $[0.002, 0.05]$ | $[0.001, 0.10]$ | $2.06\times$ | Explores orientation strategies that keep targets centered in LiDAR / sensor cones. |
| `direction_factor_min` | $[0.0, 0.50]$ | $[0.01, 1.0]$ | $1.98\times$ | **Full Holonomic Freedom:** At $1.0$, drone receives full progress reward regardless of yaw angle. |
| `retreat_penalty` | $[0.005, 0.10]$ | $[-0.01, 0.01]$ | Expanded into negative | Allows minor backward motion out of pockets without punitive penalty traps. |
| `retreat_multiplier` | $[0.5, 10.0]$ | $[0.5, 10.0]$ | $1.0\times$ | Retained for proportional scaling of continuous retreat distance. |

### 3. Impact on Population Diversity
With the expanded bounds:
1. **Hyper-Volume of Search Space:** The parameter search volume expanded by several orders of magnitude, preventing early premature convergence of genomes onto boundary clamps.
2. **True Omnidirectional Navigation:** By extending `direction_factor_min` to $1.0$, the GA can discover policies that crab, drift sideways, or reverse into target areas without wasting precious timesteps turning to face the goal.
3. **Escaping Local Minima:** The relaxed `retreat_penalty` enables the RL agent to back out of concave obstacle configurations (addressing Phase 4 U-trap dynamics) without receiving compounding penalties.

---

## 15. Statistical Rigor in Policy Evaluation (1,000 Episodes) & Phenotypic Elite Checkpoint Inheritance

### 1. The Evaluation Measurement Problem ($N = 100$)
In a bi-level reinforcement learning genetic algorithm, each candidate policy $\pi$ is scored on an evaluation set of size $N$. Let $T_i \in [0, 500]$ denote the penalized effective timesteps on episode $i$.
The sample mean estimator has Standard Error of the Mean:
$$\text{SEM} = \frac{\sigma}{\sqrt{N}}$$

In our 3-pillar arena, navigation trajectory lengths have empirical standard deviation $\sigma \approx 60\text{ steps}$.
- Under $N = 100$ episodes:
  $$\text{SEM}_{100} = \frac{60}{\sqrt{100}} = \pm 6.0\text{ steps}$$
  Because elite candidate policies in late generations differ by only $2\text{--}4\text{ steps}$ (e.g. $182.1$ vs $184.5$), a measurement noise band of $\pm 6.0\text{ steps}$ causes stochastic rank swapping in Plot 1A, obscuring true evolutionary superiority.

- Under $N = 1,000$ episodes:
  $$\text{SEM}_{1000} = \frac{60}{\sqrt{1000}} = \pm 1.89\text{ steps} \quad (\mathbf{68.4\%\text{ noise reduction!}})$$
  At $\pm 1.89\text{ steps}$, true policy distinctions dominate measurement variance, providing an unambiguous, statistically significant winner.

### 2. High-Throughput Vectorized Evaluation
Running 1,000 episodes sequentially on a single environment at $\approx 3.1\text{ eps/s}$ takes $\sim 320\text{ seconds}$ ($5.3\text{ minutes}$) per individual.
To eliminate this bottleneck, [`evaluate_policy`](file:///home/nycolas/m/evolution.py#L409) is vectorized across $20$ parallel environments using `DummyVecEnv` with pre-allocated deterministic seeds:
- Batches of 20 parallel steps execute through the Numba-accelerated raycaster at $\approx 3,800\text{ steps/s}$.
- Completed episodes are immediately replaced with the next deterministic evaluation seed ($20000 + i$).
- **Throughput:** $1,000$ episodes complete in only **$13.91\text{ seconds}$** ($\mathbf{71.9\text{ episodes/second}}$), adding negligible overhead to the evolutionary loop.

### 3. Phenotypic Elite Checkpoint Inheritance Mechanics
In classic genetic algorithms, elites represent incumbents. If an individual is an **exact clone** of its parent from the previous generation:
$$\text{is\_exact\_clone} \iff \left(\text{len}(\text{parent\_ids}) == 1\right) \land \left(\neg\,\text{is\_mutated}\right)$$

1. **Skipped Retraining:**
   Instead of resetting to [`base.zip`](file:///home/nycolas/m/base.zip) and retraining for 400k steps with stochastic SAC, the worker copies the parent's `model.zip` directly.
2. **Deterministic Metric Inheritance:**
   If the parent was already evaluated on the exact same 1,000 episodes, metrics and fitness are inherited directly ($0.0\text{s}$ compute).
3. **Selective Advantage:**
   Elites maintain perfectly horizontal lines at the top of Plot 1A. They can never regress due to bad RL training seeds. They are only displaced when a mutated or recombined child genuinely achieves a faster effective flight time.
4. **Computational Savings:**
   Saves $\sim 25\%\text{--}35\%$ of total training compute per generation, allowing the GA to complete 30 generations significantly faster.

---

## 16. Multiprocessing Pool Lifecycle & IPC Resource Tracker Management in Python 3.12

### 1. Root Cause Analysis: The 60 Leaked Semaphores
When conducting 30 generations of evolutionary search with concurrency = 2:
```
/usr/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 60 leaked semaphore objects to clean up at shutdown
```
In Python 3.8+, Python spawns an independent companion process (`resource_tracker`) responsible for tracking POSIX IPC resources (named semaphores, shared memory blocks) across processes.
- When worker processes initialize queues (`_inqueue`, `_outqueue`, `_change_notifier`), POSIX `SemLock` instances are created and registered with the resource tracker.
- Under `with ctx.Pool(processes=2) as pool:`, Python's `multiprocessing.pool.Pool.__exit__` executes:
  ```python
  def __exit__(self, exc_type, exc_val, exc_tb):
      self.terminate()
```
- Crucially, `self.terminate()` sends immediate `SIGTERM` to the worker processes and **never invokes `self.join()`**.
- Because workers are killed abruptly while communication pipes are still in transit, their cleanup handlers cannot send `UNREGISTER` commands to the `resource_tracker`.
- Over $30\text{ generations} \times 2\text{ concurrent workers} = 60\text{ workers}$, exactly $60$ semaphores remained orphaned in the tracker's registry.

### 2. The Architectural Resolution
1. **Explicit `try ... finally: pool.close(); pool.join()`:**
   - `pool.close()` transitions the pool state to `CLOSE`, signaling workers that no further tasks are queued.
   - `pool.join()` blocks until the worker handler thread, task handler thread, result handler thread, and all worker child processes terminate naturally.
2. **PyTorch File System Sharing Strategy:**
   - Setting `mp.set_sharing_strategy("file_system")` at the module root instructs PyTorch and Python IPC to back shared memory and queues with temporary disk file descriptors rather than named POSIX semaphores.
   - File descriptors are automatically reclaimed by the Linux kernel upon process death regardless of abrupt termination.
3. **Explicit CUDA IPC Garbage Collection:**
   - In [`train_individual_worker`](file:///home/nycolas/m/evolution.py#L505), `torch.cuda.ipc_collect()` and `gc.collect()` are explicitly triggered before worker completion.
4. **Resilient Formatting in Clone Inheritance:**
   - Handled `NoneType` formatting when clones fail escalating viability thresholds, preventing remote traceback worker crashes.

**Empirical Verification:** Multiple generational test cycles confirm completely clean exits with **0 leaked semaphores** and exit code 0.

---

## 17. The 19/25 Frozen Clone Anomaly & Population Diversity Mechanics

### 1. The Anomaly Observed in `ga_results_f2`
During execution with $N=25$ individuals and $n_{\text{best}} = 4$:
- Gen 00: 25 unique random individuals.
- Gen 01–29: Exactly **19 out of 25 individuals skipped retraining** every single generation, leaving only 6 individuals to train.
- Total mutated individuals per generation dropped to $0\text{--}3$, regardless of how high `--mutation_rate` was configured.

### 2. Root Cause Analysis
The population reproduction cycle divides the population into 4 groups:
- **Group 2 ($n_{\text{best}} = 4$):** Top 4 surviving elites.
- **Group 3 ($C(4, 2) = 6$):** 6 children from pairwise elite crossover.
- **Group 4 ($25 - 4 - 6 = 15$):** Remaining slots allocated to non-elite survivors via tournament selection.
- **Group 1 ($n_{\text{dead}} = 0$):** Dead replacement (allocated 0 slots because all individuals survived).

The flaw occurred because Group 4 was configured as:
```python
winner = min(tournament_sample, key=lambda ind: ind.fitness)
add_ind(copy.deepcopy(winner.genes), origin="tournament_selected", parent_ids=[winner.id], is_mutated=False)
```
1. **Exemption from Mutation:** Group 4 hardcoded `is_mutated=False`, never calling `mutate_individual()`. Increasing the mutation rate had zero effect on 15 out of 25 slots (60% of the population).
2. **False Positive in Clone Detector:** The clone detector checked `len(parent_ids) == 1 and not is_mutated`. All 15 Group 4 individuals satisfied this condition alongside the 4 elites, causing $4 + 15 = 19$ individuals to skip retraining.
3. **Loss of Diversity:** 76% of the population was frozen into unmutated copies of Generation 0 genomes.

### 3. The Structural Resolution: Tournament Recombination & Universal Mutation
1. **Tournament Crossover for Non-Elites:**
   Group 4 now selects Parent 1 via Tournament ($k=3$) from candidate survivors, Parent 2 via Tournament ($k=3$) from all survivors, and recombines them via 3-gene blend crossover.
2. **Universal Mutation Chance:**
   Every non-elite offspring across Groups 1, 3, and 4 is subjected to the dynamic mutation probability:
   $$P(\text{mutate}) = \text{current\_mutation\_rate}$$
   If triggered, 50% mutate 1 gene, 50% mutate 2 genes, perturbed by `current_mutation_band`.
3. **Strict Retraining Bypass for Elites Only:**
   ```python
   is_exact_clone = (ind.generation > 0 and ind.origin == "elite" and not ind.is_mutated)
   ```
   Non-elites (`origin != "elite"`) can never skip retraining.
4. **Guaranteed Diversity Profile:**
   - Exact Clones: Exactly $n_{\text{best}}$ (e.g. 4 or 5).
   - Retrained Offspring: Exactly $N - n_{\text{best}}$ (e.g. 21 or 20).
   - All 21 offspring feature novel crossover combinations and roll for mutation, restoring full genetic diversity.














