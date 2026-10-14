---
date: 2026-09-29
tags: [type/architecture, domain/rl-reward, status/active]
aliases: [RewardFormulation, RewardFunction]
---

# Mathematical Reward Formulation

The step reward $R_t$ at timestep $t$ in [`Drone3DEnv._computeReward`](file:///home/nycolas/m/uav_env.py) is formulated as:

$$R_t = - c_{\text{living}} - P_{\text{ground}} - P_{\text{coll}} - P_{\text{prox}} - P_{\text{fov}} - P_{\text{retreat}} + R_{\text{progress}} + R_{\text{goal}}$$

---

## Reward Component Breakdown

| Component | Mathematical Condition | Value / Formula | Behavioral Purpose |
| :--- | :--- | :--- | :--- |
| **Living Cost** | Every step | $-0.005$ | Constant time pressure discouraging hovering stagnation. |
| **Ground Penalty** | $z < 0.1\text{m}$ | $-50.0$ | Catastrophic penalty for floor crash. |
| **Physical Collision** | `_checkPhysicalCollision()` | $-50.0$ | Catastrophic penalty for striking obstacles or arena bounds. |
| **Camera Proximity** | $d_{\text{min\_sensor}} < 0.5\text{m}$ | $-(0.5 - d_{\text{min\_sensor}}) \cdot 1.0$ | Continuous repulsive potential field near obstacles. |
| **Non-FOV Motion** | $v > 0.1\text{m/s} \land \mathbf{v}_{\text{dir}} \cdot \mathbf{f}_{\text{body}} < \cos(45^\circ)$ | $-(\cos(45^\circ) - \text{alignment}) \cdot 1.0$ | Penalizes moving blind outside forward camera FOV. |
| **Progress Shaping** | Continuous | $+10.0 \cdot (d_{t-1} - d_t)$ | Potential-based progress shaping toward waypoint goal. |
| **Retreat Penalty** | $\Delta_{\text{ret}} > \delta_{\text{tol}}$ | $-k_{\text{ret}} \cdot (\Delta_{\text{ret}} - \delta_{\text{tol}})$ | Penalizes unbounded retreat while allowing local obstacle flanking. |
| **Goal Terminal** | $\|\mathbf{p}_{\text{goal}} - \mathbf{p}\| < 0.2\text{m}$ | $+100.0$ | Sparse reward for successful task completion. |

---

## 3D Multi-Step Retreat Tolerance Buffer Formulation
To bypass obstacle local minima (U-traps) without triggering infinite distance penalties:
1. Maintain a rolling ring buffer of recent goal distances $\mathcal{D}_t = [d_{t-K+1}, \dots, d_t]$ with capacity $K = 10$.
2. Find the minimum recent distance:
   $$d_{\text{min\_recent}} = \min_{d \in \mathcal{D}_t} d$$
3. Compute the current retreat magnitude:
   $$\Delta_{\text{ret}} = d_t - d_{\text{min\_recent}}$$
4. Penalize retreat beyond tolerance slack $\delta_{\text{tol}} = 0.10\text{m}$ ($k_{\text{ret}} = 2.0$):
   $$P_{\text{retreat}}(t) = \begin{cases} k_{\text{ret}} \cdot (\Delta_{\text{ret}} - \delta_{\text{tol}}), & \text{if } \Delta_{\text{ret}} > \delta_{\text{tol}} \\ 0, & \text{otherwise} \end{cases}$$

---
## Related
#domain/rl-reward #type/architecture
- [[decisions/EgocentricAlignmentAndPotentialReward]]
- [[decisions/MultiprocessVectorizationAndRetreatBuffer]]
- [[problems/SuicidePolicyExploitation]]
- [[problems/ObstacleUTrapDeadlock]]
- [[architecture/EnvironmentArchitecture]]
