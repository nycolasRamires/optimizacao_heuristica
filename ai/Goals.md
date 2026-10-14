---
date: 2026-09-29
tags: [project/goals, domain/planning, status/active]
aliases: [goals, ProjectGoals, Roadmap]
---

# Project Goals & Progress

Central roadmap and milestone tracker for the 3D Quadrotor Autonomous Navigation and Evolutionary Reinforcement Learning campaign.

---

## Milestone Checklist

- [x] **Phase 1: 3D Environment Architecture & Diagnostics**
  - [x] Base 3D quadrotor simulation in PyBullet (`Drone3DEnv` in [`uav_env.py`](file:///home/nycolas/m/uav_env.py)) — [[decisions/3DSimulationPlatform]], [[architecture/EnvironmentArchitecture]]
  - [x] 4 randomly placed cylindrical obstacles ($r=0.15\text{m}, h=2.0\text{m}$) and obstacle clearance verification — [[architecture/EnvironmentArchitecture]]
  - [x] 64x64 depth camera observation and 23D kinematic state vector — [[decisions/MultiModalPerception]], [[architecture/PerceptionPipeline]]
  - [x] Absolute waypoint action space with inner-loop PID controller ($40\text{ Hz}$ control / $200\text{ Hz}$ physics) — [[decisions/CascadedControlHierarchy]]
  - [x] PyBullet contact manifold ground-truth collision checking (`_checkPhysicalCollision`) — [[decisions/ContactManifoldCollisionChecking]]
  - [x] SAC baseline training pipeline with `MultiInputPolicy` ([`train.py`](file:///home/nycolas/m/train.py))
  - [x] Interactive 3D OpenGL GUI visualizer and HD video recorder ([`enjoy.py`](file:///home/nycolas/m/enjoy.py))
  - [x] Eliminate "suicide policy" loophole: balanced collision/ground penalties ($-50.0$) and implemented potential progress shaping — [[problems/SuicidePolicyExploitation]], [[decisions/EgocentricAlignmentAndPotentialReward]], [[architecture/RewardFormulation]]
  - [x] Fix depth camera contrast: normalized linear metric depth across full $[0, 255]$ dynamic range — [[problems/DepthBufferFlattening]], [[decisions/EgocentricAlignmentAndPotentialReward]], [[architecture/PerceptionPipeline]]
  - [x] Standardize coordinate frames: rotate relative target displacement into drone's egocentric body frame — [[problems/WorldFrameGoalMisalignment]], [[decisions/EgocentricAlignmentAndPotentialReward]], [[architecture/PerceptionPipeline]]
  - [x] Address depth camera rendering throughput bottleneck (`ER_NO_SEGMENTATION_MASK` + `SubprocVecEnv`) — [[problems/CPURenderingBottleneck]], [[decisions/MultiprocessVectorizationAndRetreatBuffer]], [[architecture/HighThroughputProfiling]]
  - [x] Implement multi-step 3D retreat tolerance buffer ($\delta_{\text{tol}} = 0.10\text{m}$, $k_{\text{ret}} = 2.0$) to resolve obstacle U-traps — [[problems/ObstacleUTrapDeadlock]], [[decisions/MultiprocessVectorizationAndRetreatBuffer]], [[architecture/RewardFormulation]]
  - [ ] Implement parameterizable reward dictionary for automated GA reward tuning

- [ ] **Phase 2: High-Throughput Vectorization & Baseline Solidification**
  - [x] Benchmark simulation step throughput (FPS) and SAC training speed (steps/s) — [[architecture/HighThroughputProfiling]]
  - [x] Parallelize PyBullet via `SubprocVecEnv(12)` achieving $589.8\text{ steps/s}$ ($7.3\times$ speedup) — [[decisions/MultiprocessVectorizationAndRetreatBuffer]]
  - [ ] Pre-train solid 3D baseline checkpoint (`base_3d.zip`)
  - [ ] Benchmark baseline deterministic success, collision, and timeout rates over 100 test episodes

- [ ] **Phase 3: 3D Genetic Algorithm Optimization**
  - [ ] Formulate 3D evolvable genome (altitude buffers, tilt penalties, obstacle clearances, speed shaping)
  - [ ] Implement multi-tier reproduction cycle with strict tier isolation (Elite × Elite, Non-Elite × Non-Elite)
  - [ ] Implement dynamic viability threshold and penalized effective timesteps fitness
  - [ ] Run 3D evolutionary search campaign
  - [ ] Generate uncluttered 3D evolutionary visual analytics dashboard

---
## Related
#domain/planning #project/goals #status/active
- [[Index]]
- [[Decisions]]
- [[Explanations]]
- [[architecture/EnvironmentArchitecture]]
- [[architecture/RewardFormulation]]
