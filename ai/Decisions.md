---
date: 2026-09-29
tags: [index, type/decision, domain/architecture]
aliases: [decisions, DecisionsLog, EngineeringDecisions]
---

# Project Engineering Decisions Log

This central log records major architectural pivots, trade-offs, and critical design choices. Detailed context, root-cause investigations, and mathematical proofs are tracked in dedicated decision records.

---

## Architectural Decision Records (ADRs)

### [[decisions/3DSimulationPlatform|Decision 1: PyBullet 3D Simulation Platform & Crazyflie 2.X Quadrotor Model]]
- **Context:** Transitioning from 2D planar navigation to 3D flight requires realistic 6-DOF physics, gravity, rotor dynamics, and obstacle contact dynamics.
- **Decision:** Use `gym-pybullet-drones` with Crazyflie 2.X (`DroneModel.CF2X`) and PyBullet physics backend (`Physics.PYB`).
- **Rationale:** Verified rigid-body aerodynamic parameters, ground contact simulation, and native collision geometries without custom engine overhead.
- **Traceability:** Addresses [[problems/3DSimulationPlatform]] via [[investigations/3DSimulationPlatform]].

### [[decisions/CascadedControlHierarchy|Decision 2: Cascaded Control Hierarchy (3D Waypoint Action Space + Inner PID)]]
- **Context:** Training RL agents directly on raw 4-motor RPMs results in tumbling and slow policy exploration.
- **Decision:** Expose a 3D absolute spatial waypoint action space $[x, y, z]$ with an inner-loop PID controller converting position errors into motor commands at $40\text{ Hz}$ control / $200\text{ Hz}$ physics.
- **Rationale:** Ensures flight stability from step 0, allowing the agent to focus on path planning rather than low-level stabilization.
- **Traceability:** Addresses [[problems/MotorControlInstability]] via [[investigations/CascadedControlHierarchy]].

### [[decisions/MultiModalPerception|Decision 3: Multi-Modal Perception (64x64 Forward Depth Image + 23D Kinematics)]]
- **Context:** Safe navigation around unmapped 3D obstacles requires both spatial vision and precise body state estimation.
- **Decision:** Adopt a dictionary observation space in SB3 with `MultiInputPolicy`: $1 \times 64 \times 64$ normalized depth image and 23D kinematic state vector.
- **Rationale:** Combines rich spatial depth for obstacle avoidance with high-precision proprioceptive feedback for stable motion control.
- **Traceability:** Addresses [[problems/Unmapped3DObstaclePerception]] via [[investigations/MultiModalPerception]].

### [[decisions/ContactManifoldCollisionChecking|Decision 4: Contact Manifold Ground-Truth Collision Checking]]
- **Context:** A forward camera with $90^\circ$ FOV cannot detect obstacles that hit the drone from the side, rear, top, or bottom.
- **Decision:** Query PyBullet's contact points (`p.getContactPoints`) to detect true physical collisions against obstacles and the floor.
- **Rationale:** Prevents blind-spot exploitation where the drone turns away and clips obstacles with rear propellers without penalty.
- **Traceability:** Addresses [[problems/CameraBlindSpotCollisions]] via [[investigations/ContactManifoldCollisionChecking]].

### [[decisions/EgocentricAlignmentAndPotentialReward|Decision 5: Egocentric Multi-Modal Alignment & Potential-Based Reward Formulation]]
- **Context:** Early trials suffered from a "suicide policy" loophole (low collision penalty vs static distance bleed), raw Z-buffer contrast compression, and world-coordinate goal vectors.
- **Decision:** Replace static distance bleed with potential progress shaping ($+10.0 \cdot \Delta d$), raise collision/ground penalty to $-50.0$, normalize metric linear depth, and rotate relative target into the body frame ($\mathbf{R}^T \Delta\mathbf{p}$).
- **Rationale:** Eliminates the suicide loop, restores CNN obstacle visual contrast, and provides rotation-invariant egocentric consistency.
- **Traceability:** Addresses [[problems/SuicidePolicyExploitation]], [[problems/DepthBufferFlattening]], and [[problems/WorldFrameGoalMisalignment]] via [[investigations/SensoryAndRewardAlignment]].

### [[decisions/MultiprocessVectorizationAndRetreatBuffer|Decision 6: Multiprocess Vectorization & 3D Multi-Step Retreat Buffer]]
- **Context:** Software camera rendering bound single-process throughput to $80.5\text{ steps/s}$, and static progress penalties caused deadlock in obstacle U-traps.
- **Decision:** Distribute simulation across 12 CPU workers via `SubprocVecEnv` with pipelined training (`train_freq=(4, 'step')`, `gradient_steps=4`, `batch_size=256`), disabling segmentation masks (`ER_NO_SEGMENTATION_MASK`), and implement a 10-step rolling distance retreat buffer ($\delta_{\text{tol}} = 0.10\text{m}$, $k_{\text{ret}} = 2.0$).
- **Rationale:** Accelerates training throughput by $7.3\times$ to $\sim 590\text{ steps/s}$ while granting the quadrotor sufficient geometric slack to flank pillars without local-minima entrapment.
- **Traceability:** Addresses [[problems/CPURenderingBottleneck]] and [[problems/ObstacleUTrapDeadlock]] via [[investigations/VectorizedThroughputAndRetreatBuffer]].

### [[decisions/DenseObstacleForestAndWarmStart|Decision 7: Dense 8-Pillar Forest Environment & Warm-Started Policy Curriculum (navDrone_v2)]]
- **Context:** In the 4-pillar arena, sparse obstacle placement allowed straight-line flight paths to the target without requiring active obstacle avoidance.
- **Decision:** Increase obstacle count to 8 cylindrical pillars with radius $0.20\text{m}$ (diameter $40\text{cm}$), enforce $0.55\text{m}$ pairwise/target clearances, and warm-start `navDrone_v2` training directly from `navDrone_v1` weights and normalization statistics.
- **Rationale:** Forces reactive 3D obstacle avoidance across dense flight corridors while retaining pre-trained waypoint tracking stability.

---
## Related
#domain/architecture #type/decision #index
- [[Index]]
- [[Goals]]
- [[Explanations]]
- [[architecture/EnvironmentArchitecture]]
- [[architecture/PerceptionPipeline]]
- [[architecture/RewardFormulation]]
- [[architecture/HighThroughputProfiling]]
