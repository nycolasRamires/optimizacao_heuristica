---
date: 2026-09-29
tags: [type/problem, domain/simulation, status/resolved]
aliases: [CameraBlindSpotCollisionsProblem]
---

> Resolved by: [[decisions/ContactManifoldCollisionChecking]]

# Problem: Camera Blind-Spot Collisions & Boundary Clipping

## Context & Requirement
In an autonomous quadrotor environment equipped with a single forward-facing camera ($90^\circ$ Field of View), the agent has full vision of obstacles ahead, but zero direct vision to its rear, flanks, top, and bottom.

## Symptoms & Constraints
- **Policy Exploitation:** When evaluating collisions purely based on visual obstacle proximity or forward depth thresholds, the policy learns to turn away from obstacles or reverse laterally into pillars.
- **Physical Inaccuracy:** Propellers and chassis clip through obstacles without triggering collision termination if the collision point falls outside the camera frustum.
- **Unpenalized Damage:** A drone rotating rapidly could sideswipe a cylinder with its rear arm without incurring a collision penalty, violating real-world safety guarantees.

---
## Related
#domain/simulation #type/problem #status/resolved
- [[investigations/ContactManifoldCollisionChecking]]
- [[decisions/ContactManifoldCollisionChecking]]
- [[architecture/EnvironmentArchitecture]]
