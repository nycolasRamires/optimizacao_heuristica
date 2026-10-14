---
date: 2026-09-29
tags: [type/problem, domain/rl-navigation, status/resolved]
aliases: [ObstacleUTrapDeadlockProblem]
---

> Resolved by: [[decisions/MultiprocessVectorizationAndRetreatBuffer]]

# Problem: Obstacle U-Trap Deadlock & Flanking Inability

## Context & Requirement
In 3D cluttered navigation, cylindrical pillars are frequently placed directly along the direct line of sight between the drone and the target waypoint. The policy must learn to perform lateral circumnavigation or temporary retreat maneuvers to bypass obstacles.

## Symptoms & Diagnosis
- **Monotonic Distance Penalization:** With strict potential shaping or distance penalties, every single step that increases distance to the goal ($d_t > d_{t-1}$) is immediately penalized.
- **Deadlock in U-Traps:** When a drone flies toward a pillar and reaches minimum obstacle clearance, turning around or taking a wide lateral arc necessarily increases distance to the target for several timesteps.
- **Hovering Stagnation:** The policy chooses to hover in place directly in front of the pillar to avoid increasing distance, resulting in episode timeouts ($t = 600$ steps) without reaching the goal.

---
## Related
#domain/rl-navigation #type/problem #status/resolved
- [[investigations/VectorizedThroughputAndRetreatBuffer]]
- [[decisions/MultiprocessVectorizationAndRetreatBuffer]]
- [[architecture/RewardFormulation]]
