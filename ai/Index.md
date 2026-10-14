---
date: 2026-09-29
tags: [index, domain/architecture, status/active]
aliases: [Index, Home, VaultOverview]
---

# 3D Quadrotor Autonomous Navigation Knowledge Base

Welcome to the central Obsidian knowledge graph for the 3D Quadrotor Autonomous Navigation and Evolutionary Reinforcement Learning project.

---

## Central Maps of Content (MOCs)
- **[[Goals]]:** Project roadmap, phase tracking, and completed milestones.
- **[[Decisions]]:** Architectural Decision Records (ADR) log documenting major pivots and trade-offs.
- **[[Explanations]]:** Technical explanations, system dynamics, and mathematical foundations.

---

## Core Architecture Specifications
- **[[architecture/EnvironmentArchitecture]]:** PyBullet physics engine, Crazyflie 2.X dynamics, and control loop hierarchy.
- **[[architecture/PerceptionPipeline]]:** 23D kinematic state vector, forward depth camera pipeline, and metric linearization.
- **[[architecture/RewardFormulation]]:** Mathematical reward formulation, potential shaping, and retreat buffer.
- **[[architecture/HighThroughputProfiling]]:** Micro-benchmarks, bottleneck analysis, and multiprocess vectorization.

---

## Knowledge Traceability Graph (Problems $\rightarrow$ Investigations $\rightarrow$ Decisions)

```
[Problems] ───────────────► [Investigations] ───────────────► [Decisions]
```

| Domain | Problem | Investigation | Decision Record |
| :--- | :--- | :--- | :--- |
| **Simulation** | [[problems/3DSimulationPlatform]] | [[investigations/3DSimulationPlatform]] | [[decisions/3DSimulationPlatform]] |
| **Control** | [[problems/MotorControlInstability]] | [[investigations/CascadedControlHierarchy]] | [[decisions/CascadedControlHierarchy]] |
| **Perception** | [[problems/Unmapped3DObstaclePerception]] | [[investigations/MultiModalPerception]] | [[decisions/MultiModalPerception]] |
| **Collision** | [[problems/CameraBlindSpotCollisions]] | [[investigations/ContactManifoldCollisionChecking]] | [[decisions/ContactManifoldCollisionChecking]] |
| **Reward & Sensory** | [[problems/SuicidePolicyExploitation]]<br>[[problems/DepthBufferFlattening]]<br>[[problems/WorldFrameGoalMisalignment]] | [[investigations/SensoryAndRewardAlignment]] | [[decisions/EgocentricAlignmentAndPotentialReward]] |
| **Throughput & Navigation** | [[problems/CPURenderingBottleneck]]<br>[[problems/ObstacleUTrapDeadlock]] | [[investigations/VectorizedThroughputAndRetreatBuffer]] | [[decisions/MultiprocessVectorizationAndRetreatBuffer]] |

---
## Related
#domain/architecture #index
- [[Goals]]
- [[Decisions]]
- [[Explanations]]
