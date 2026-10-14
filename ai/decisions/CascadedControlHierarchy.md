---
date: 2026-09-29
tags: [type/decision, domain/control, status/resolved]
aliases: [Decision2, Decision-002, DecisionCascadedControl]
---

> Resolves: [[investigations/CascadedControlHierarchy]]
> Related to: [[problems/MotorControlInstability]]

# Decision: Cascaded Control Hierarchy (3D Waypoint Action Space + Inner PID)

## Context
Training RL agents directly on raw 4-motor RPMs often results in unstable tumbling, erratic crashes, and exponentially slow policy exploration.

## Decision
Expose a 3D absolute spatial waypoint action space $[x, y, z] \in [-2.5, 2.5] \times [-2.5, 2.5] \times [0.0, 2.5]$ to the RL policy, while an inner-loop PID controller (`DSLPIDControl`) converts position errors into motor commands at $40\text{ Hz}$ control / $200\text{ Hz}$ physics.

## Rationale
- **Inherent Flight Stability:** Drone achieves stable hovering and bounded trajectories from step 0 of training.
- **Sample Efficiency:** The policy exploration space is reduced from high-dimensional aerodynamic torques to clean 3D geometric navigation.
- **Intermediate Slicing:** A step displacement clamp (`step_size=1.0m` in `_calculateNextStep`) prevents infeasible step jumps and maintains smooth trajectories.

---
## Related
#domain/control #type/decision #status/resolved
- [[problems/MotorControlInstability]]
- [[investigations/CascadedControlHierarchy]]
- [[architecture/EnvironmentArchitecture]]
- [[Decisions]]
