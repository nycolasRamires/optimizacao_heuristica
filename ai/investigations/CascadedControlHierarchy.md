---
date: 2026-09-29
tags: [type/investigation, domain/control, status/resolved]
aliases: [CascadedControlHierarchyInvestigation]
---

> Addresses: [[problems/MotorControlInstability]]
> Resolved by: [[decisions/CascadedControlHierarchy]]

# Investigation: Action Space Hierarchies for 3D Quadrotor Navigation

## Control Paradigms Analyzed

### 1. Direct Motor RPM Commands ($\mathbb{R}^4$)
- **Action:** Desired RPM for 4 brushless motors.
- **Evaluation:** High degree of freedom, but learning curves show extreme variance. Over 99% of early exploration episodes fail immediately due to aerodynamic tumbling.

### 2. Low-Level Attitude Rate Control (Body Rates $\omega_x, \omega_y, \omega_z$ + Collective Thrust $T$)
- **Action:** Angular velocity vector and normalized collective thrust.
- **Evaluation:** Mitigates rotor-level imbalance, but policy still struggles to maintain hover or translate smoothly between obstacle waypoints without drift.

### 3. Cascaded Hierarchical Control (Absolute Waypoint Action Space + Inner PID)
- **Action:** Desired 3D spatial coordinate $\mathbf{p}_{\text{cmd}} = [x_{\text{cmd}}, y_{\text{cmd}}, z_{\text{cmd}}] \in \mathbb{R}^3$.
- **Mechanism:** An intermediate slicer bounds the step displacement ($\Delta p \le 1.0\text{m}$), and a deterministic inner-loop PID controller (`DSLPIDControl`) converts position and velocity errors into 4 motor RPM commands at $40\text{ Hz}$ control frequency ($200\text{ Hz}$ physics).
- **Evaluation:** Provides immediate flight stability from step 0. Completely decouples aerodynamic attitude stabilization from obstacle avoidance and path planning.

## Outcome
Adopting cascaded control ensures reliable flight dynamics, allowing reinforcement learning algorithms to focus entirely on path optimization and dynamic obstacle circumvention.

---
## Related
#domain/control #type/investigation #status/resolved
- [[problems/MotorControlInstability]]
- [[decisions/CascadedControlHierarchy]]
- [[architecture/EnvironmentArchitecture]]
