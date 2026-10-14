---
date: 2026-09-29
tags: [type/investigation, domain/simulation, status/resolved]
aliases: [3DSimulationPlatformInvestigation]
---

> Addresses: [[problems/3DSimulationPlatform]]
> Resolved by: [[decisions/3DSimulationPlatform]]

# Investigation: Evaluation of 3D Quadrotor Simulation Backends

## Options Evaluated

### Option 1: Custom 3D Rigid-Body Physics Engine (NumPy / SciPy)
- **Pros:** Zero external C-library dependency; direct low-level control of mathematical equations of motion.
- **Cons:** High engineering overhead; complex collision mesh mathematics; lack of verified real-world aerodynamic parameterizations and ground contact physics.

### Option 2: Isaac Gym / Isaac Sim
- **Pros:** Massively parallel GPU-accelerated environments ($10,000+$ envs on a single workstation).
- **Cons:** Complex installation footprint; proprietary dependencies; high complexity for custom onboard vision depth synthesis and small-scale prototyping.

### Option 3: PyBullet via `gym-pybullet-drones`
- **Pros:** 
  - Native C++ collision detection via contact manifolds (`p.getContactPoints`).
  - Verified Crazyflie 2.X aerodynamic parameters ($m = 27\text{ g}$, $T/W = 2.25$, drag coefficients).
  - Lightweight headless execution with TinyRenderer support.
  - Native compatibility with Gymnasium and Stable-Baselines3.
- **Cons:** CPU-bound depth rendering if unoptimized; requires multiprocess vectorization (`SubprocVecEnv`) for high training throughput.

## Findings & Outcome
`gym-pybullet-drones` provides the optimal balance of aerodynamic realism, built-in contact checking, and modular extensibility for deep RL.

---
## Related
#domain/simulation #type/investigation #status/resolved
- [[problems/3DSimulationPlatform]]
- [[decisions/3DSimulationPlatform]]
- [[architecture/EnvironmentArchitecture]]
