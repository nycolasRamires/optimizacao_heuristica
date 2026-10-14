---
date: 2026-09-29
tags: [type/problem, domain/perception, status/resolved]
aliases: [Unmapped3DObstaclePerceptionProblem]
---

> Resolved by: [[decisions/MultiModalPerception]]

# Problem: Perception of Unmapped Obstacles in 3D Flight

## Context & Requirement
Navigating an autonomous drone through an environment with randomly distributed, unmapped 3D obstacles requires both spatial vision (geometry, clearance, relative depth) and high-fidelity proprioceptive state estimation (current velocity, orientation, motor dynamics, and goal bearing).

## Symptoms & Constraints
- **Vision-Only Limitations:** Raw vision lacks direct velocity and acceleration sensing, causing latency in attitude control and trajectory overshoot.
- **State-Only Limitations:** Relying solely on proprioceptive kinematics leaves the agent blind to unmapped obstacles, making real-time collision avoidance mathematically impossible without pre-built global maps.
- **Algorithm Architecture:** Standard MLP policies cannot process visual grids, while standard CNNs cannot naturally digest arbitrary kinematic state vectors without specialized multi-head feature extractors.

---
## Related
#domain/perception #type/problem #status/resolved
- [[investigations/MultiModalPerception]]
- [[decisions/MultiModalPerception]]
- [[architecture/PerceptionPipeline]]
