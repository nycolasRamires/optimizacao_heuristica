---
date: 2026-09-29
tags: [type/problem, domain/perception, status/resolved]
aliases: [WorldFrameGoalMisalignmentProblem]
---

> Resolved by: [[decisions/EgocentricAlignmentAndPotentialReward]]

# Problem: World Coordinates in Relative Goal Observation

## Context & Requirement
In multi-modal navigation, proprioceptive observations and exteroceptive visual observations must share a coherent coordinate reference frame or allow straightforward spatial correspondence.

## Symptoms & Diagnosis
- The visual depth observation is rendered from the drone's forward body camera, meaning that image pixels directly correspond to the drone's **egocentric reference frame** (forward, left, up).
- However, the relative goal vector was computed as:
  $$\mathbf{p}_{\text{target, rel}} = \mathbf{p}_{\text{goal}} - \mathbf{p}_{\text{drone}} \in \mathbb{R}^3$$
  which is expressed strictly in **inertial world coordinates**.

### Consequence
When the drone turned (e.g., yaw $= 90^\circ$ East), an obstacle visible directly ahead in the forward depth map corresponded to an obstacle located East in the world frame, while a world vector pointing North required an action to the drone's left. The policy network was forced to internally discover and compute a 3D rotation matrix from raw attitude quaternions to reconcile the camera image with the goal displacement, severely degrading policy learning speed.

---
## Related
#domain/perception #type/problem #status/resolved
- [[investigations/SensoryAndRewardAlignment]]
- [[decisions/EgocentricAlignmentAndPotentialReward]]
- [[architecture/PerceptionPipeline]]
