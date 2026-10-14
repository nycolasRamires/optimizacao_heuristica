---
date: 2026-09-29
tags: [type/problem, domain/control, status/resolved]
aliases: [MotorControlInstabilityProblem]
---

> Resolved by: [[decisions/CascadedControlHierarchy]]

# Problem: Quadrotor Motor Control Instability in Reinforcement Learning

## Context & Requirement
In multi-rotor flight control, the raw actuation inputs are the individual motor speeds ($\Omega_1, \Omega_2, \Omega_3, \Omega_4$ in RPM). Training a reinforcement learning policy directly on motor RPM commands requires the policy to simultaneously learn basic aerodynamic attitude stabilization, gyroscopic torque balancing, and high-level 3D spatial navigation.

## Symptoms & Constraints
- **Uncontrolled Tumbling:** Early policy exploration generates asymmetric RPMs that induce rapid pitch/roll divergence, causing immediate ground crashes within $1\text{--}5$ simulation steps.
- **Exploration Sample Inefficiency:** The probability of randomly exploring an RPM combination that keeps the quadrotor stably hovering is exponentially small ($\ll 1\%$).
- **Objective Interference:** The reward gradient for obstacle circumvention is completely overwhelmed by high-frequency attitude instability penalties.

---
## Related
#domain/control #type/problem #status/resolved
- [[investigations/CascadedControlHierarchy]]
- [[decisions/CascadedControlHierarchy]]
- [[architecture/EnvironmentArchitecture]]
