---
date: 2026-09-29
tags: [type/decision, domain/perception, status/resolved]
aliases: [Decision3, Decision-003, DecisionMultiModalPerception]
---

> Resolves: [[investigations/MultiModalPerception]]
> Related to: [[problems/Unmapped3DObstaclePerception]]

# Decision: Multi-Modal Perception (64x64 Forward Depth Image + 23D Kinematics)

## Context
Navigating around unmapped 3D obstacles requires both spatial vision and precise body state estimation for attitude and motion control.

## Decision
Adopt a dictionary observation space in Stable-Baselines3 using `MultiInputPolicy`:
1. `"image"`: $1 \times 64 \times 64$ normalized depth image from a forward-facing camera rendered in PyBullet.
2. `"state"`: 23D vector containing position, orientation quaternion, Euler angles, linear/angular velocities, motor speeds, and egocentric relative goal displacement.

## Rationale
- **Spatial Collision Avoidance:** The $64 \times 64$ depth map provides continuous distance cues to cylindrical obstacles and arena boundaries directly in front of the flight path.
- **Flight Stability:** The 23D vector supplies the actor with full rigid-body state information, allowing accurate speed and heading corrections without temporal inference lag.
- **SB3 Integration:** Natively supported by `MultiInputPolicy`, passing vision through a CNN and kinematics through an MLP trunk before actor-critic heads.

---
## Related
#domain/perception #type/decision #status/resolved
- [[problems/Unmapped3DObstaclePerception]]
- [[investigations/MultiModalPerception]]
- [[architecture/PerceptionPipeline]]
- [[Decisions]]
