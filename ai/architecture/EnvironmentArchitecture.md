---
date: 2026-09-29
tags: [type/architecture, domain/simulation, status/active]
aliases: [EnvironmentArchitecture, 3DEnvArchitecture]
---

# 3D Quadrotor Environment Architecture

## System Diagram
The 3D environment ([`uav_env.py`](file:///home/nycolas/m/uav_env.py)) builds upon `gym-pybullet-drones` (`BaseRLAviary`), simulating a **Crazyflie 2.X** quadrotor in a $5\text{m} \times 5\text{m} \times 2.5\text{m}$ operational volume with dynamic obstacles:

```
                      [Higher-Level RL Policy (SAC)]
                                     │
                      Action: Absolute Waypoint [x, y, z] (40 Hz)
                                     ▼
                      [Intermediate Waypoint Slicer]
                       _calculateNextStep(step_size=1.0m)
                                     ▼
                      [Inner-Loop DSLPIDControl (40 Hz)]
                                     │
                      4 Motor RPM Commands [rpm1, rpm2, rpm3, rpm4]
                                     ▼
                      [PyBullet Physics Engine (200 Hz)]
                       (5 sub-steps per environment step)
                                     │
              ┌──────────────────────┴──────────────────────┐
              ▼                                             ▼
       [Contact Manifold]                          [Camera Renderer]
   Ground Truth Collision Checking          64x64 Depth Image (ER_TINY_RENDERER)
```

## Simulation & Control Parameters
- **Physics Backend:** PyBullet (`Physics.PYB`) running at $200\text{ Hz}$ (`pyb_freq=200`, $5\text{ ms}$ timestep).
- **Control Frequency:** $40\text{ Hz}$ (`ctrl_freq=40`, $25\text{ ms}$ per environment step). Each environment step executes exactly 5 physics sub-steps.
- **Drone Dynamics:** Crazyflie 2.X ($m = 27\text{ g}$, arm length $L = 3.97\text{ cm}$, thrust-to-weight ratio $T/W = 2.25$).
- **Arena & Obstacles:**
  - Ground plane at $z = 0$.
  - 4 static vertical cylindrical pillars ($r = 0.15\text{m}$, height $h = 2.0\text{m}$) spawned randomly in $[-1.5, 1.5] \times [-1.5, 1.5]$ with $\ge 0.5\text{m}$ clearance from drone and target.
  - Target: Green visual sphere ($r = 0.15\text{m}$) spawned in $[-2.0, 2.0] \times [-2.0, 2.0] \times [0.5, 2.0]$.

## Action Space Specification
- RL Action Vector: $\mathbf{a} \in [-1, 1]^3$, mapped to spatial waypoints:
  $$x_{\text{cmd}}, y_{\text{cmd}} \in [-2.5, 2.5]\text{m}, \quad z_{\text{cmd}} \in [0.0, 2.5]\text{m}$$
- Step Slicing: `_calculateNextStep` limits the maximum target shift to $1.0\text{m}$ per step.
- Inner Loop: `DSLPIDControl` generates 4 individual propeller RPMs to track position and yaw setpoints.

---
## Related
#domain/simulation #type/architecture
- [[decisions/3DSimulationPlatform]]
- [[decisions/CascadedControlHierarchy]]
- [[decisions/ContactManifoldCollisionChecking]]
- [[architecture/PerceptionPipeline]]
- [[architecture/RewardFormulation]]
- [[architecture/HighThroughputProfiling]]
