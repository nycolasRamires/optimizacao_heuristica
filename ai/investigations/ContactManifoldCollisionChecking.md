---
date: 2026-09-29
tags: [type/investigation, domain/simulation, status/resolved]
aliases: [ContactManifoldCollisionCheckingInvestigation]
---

> Addresses: [[problems/CameraBlindSpotCollisions]]
> Resolved by: [[decisions/ContactManifoldCollisionChecking]]

# Investigation: Collision Checking Mechanisms in PyBullet 3D Simulation

## Mechanisms Evaluated

### 1. Vision Frustum Proximity Thresholding
- **Mechanism:** Flag collision if forward depth pixels drop below minimum threshold ($d_{\min} < 0.15\text{m}$).
- **Drawbacks:** Completely blind to sides, rear, top, and bottom. Policies learn evasive yaw rotation exploits that obscure obstacle proximity while colliding physically.

### 2. Analytical Distance Checking (Geometric Cylinder/Sphere Distance)
- **Mechanism:** Compute analytical Euclidean distance between drone centroid $\mathbf{p}_{\text{drone}}$ and all obstacle cylinders $(x_i, y_i, r)$.
- **Drawbacks:** Ignores attitude orientation and arm length. A pitched or rolled drone hits obstacles earlier than a flat drone; does not generalize if non-primitive obstacle meshes are added.

### 3. Physics Engine Contact Manifolds (`p.getContactPoints`)
- **Mechanism:** Query PyBullet internal collision detector directly:
  ```python
  pts = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT)
  ```
- **Evaluation:** PyBullet computes GJK/EPA continuous and discrete narrow-phase contact manifolds across all rigid bodies (obstacles, arena ground plane, walls).
- **Execution Cost:** Highly optimized C++ routine; micro-profiling demonstrates execution time is $< 0.001\text{ ms}$ per step ($0.03\%$ of environment step time).

## Outcome
Querying PyBullet's native contact manifold provides exact, robust, and omnidirectional ground-truth collision detection with virtually zero computational overhead.

---
## Related
#domain/simulation #type/investigation #status/resolved
- [[problems/CameraBlindSpotCollisions]]
- [[decisions/ContactManifoldCollisionChecking]]
- [[architecture/EnvironmentArchitecture]]
