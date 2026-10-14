---
date: 2026-09-29
tags: [type/decision, domain/simulation, status/resolved]
aliases: [Decision4, Decision-004, DecisionContactManifold]
---

> Resolves: [[investigations/ContactManifoldCollisionChecking]]
> Related to: [[problems/CameraBlindSpotCollisions]]

# Decision: Contact Manifold Ground-Truth Collision Checking

## Context
A forward-facing camera with a $90^\circ$ FOV cannot detect obstacles that hit the drone from the side, rear, top, or bottom. Relying on visual proximity creates blind-spot policy exploits.

## Decision
Query PyBullet's native C++ contact point manifolds (`p.getContactPoints`) via `_checkPhysicalCollision()` to detect physical contact against obstacle pillars and the ground plane, completely independent of camera frustum orientation.

## Rationale
- **Zero Blind Spots:** Detects physical contact across all drone linkages, arms, and rotors regardless of drone attitude or yaw angle.
- **Microsecond Execution Speed:** Directly queries PyBullet's internal narrow-phase collision hash table ($< 0.001\text{ ms}$ per step), adding negligible overhead.
- **Robust Reward Signaling:** Enables unequivocal ground-truth termination and penalty assignment, enforcing safe flight margins.

---
## Related
#domain/simulation #type/decision #status/resolved
- [[problems/CameraBlindSpotCollisions]]
- [[investigations/ContactManifoldCollisionChecking]]
- [[architecture/EnvironmentArchitecture]]
- [[Decisions]]
