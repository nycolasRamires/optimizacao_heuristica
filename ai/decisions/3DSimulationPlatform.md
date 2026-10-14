---
date: 2026-09-29
tags: [type/decision, domain/simulation, status/resolved]
aliases: [Decision1, Decision-001, DecisionPyBulletPlatform]
---

> Resolves: [[investigations/3DSimulationPlatform]]
> Related to: [[problems/3DSimulationPlatform]]

# Decision: PyBullet 3D Simulation Platform & Crazyflie 2.X Quadrotor Model

## Context
Transitioning from 2D planar navigation to 3D autonomous flight requires realistic 6-DOF physics, gravity, rotor dynamics, and obstacle contact dynamics without the overhead of building a custom physics simulator.

## Decision
Adopt `gym-pybullet-drones` (`BaseRLAviary`) using the **Crazyflie 2.X** quadrotor model (`DroneModel.CF2X`) and PyBullet's physics backend (`Physics.PYB`) running at $200\text{ Hz}$ physics and $40\text{ Hz}$ control frequency.

## Rationale
- **Aerodynamic Fidelity:** Implements verified physical properties ($m = 27\text{ g}$, arm length $L = 3.97\text{ cm}$, thrust-to-weight ratio $T/W = 2.25$).
- **Deterministic Contact Manifolds:** PyBullet native C++ collision checking detects ground-truth obstacle strikes.
- **Gymnasium Ecosystem:** Seamless integration with Stable-Baselines3 SAC algorithms and vectorized wrappers.

---
## Related
#domain/simulation #type/decision #status/resolved
- [[problems/3DSimulationPlatform]]
- [[investigations/3DSimulationPlatform]]
- [[architecture/EnvironmentArchitecture]]
- [[Decisions]]
