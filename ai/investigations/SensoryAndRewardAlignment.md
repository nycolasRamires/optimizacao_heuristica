---
date: 2026-09-29
tags: [type/investigation, domain/rl-reward, domain/perception, status/resolved]
aliases: [SensoryAndRewardAlignmentInvestigation]
---

> Addresses: [[problems/SuicidePolicyExploitation]], [[problems/DepthBufferFlattening]], [[problems/WorldFrameGoalMisalignment]]
> Resolved by: [[decisions/EgocentricAlignmentAndPotentialReward]]

# Investigation: Sensory Representation & Reward Formulation Co-Design

## Scope of Investigation
This investigation analyzed the triad of behavioral and perceptual pathologies observed during early Phase 1 3D flight trials:
1. "Suicide policy" exploitation ([[problems/SuicidePolicyExploitation]]).
2. Non-linear depth buffer contrast loss ([[problems/DepthBufferFlattening]]).
3. Coordinate frame misalignment between vision and kinematics ([[problems/WorldFrameGoalMisalignment]]).

---

## 1. Reward Function Restructuring

### Analysis of Potential-Based Reward Shaping (PBRS)
To eliminate the suicide policy without sacrificing continuous progress signals:
- Static distance penalties ($-c \cdot d_t$) penalize flight duration rather than lack of progress.
- In contrast, potential-based progress shaping evaluates:
  $$\Delta d_t = d_{t-1} - d_t$$
  Where $d_t = \|\mathbf{p}_{\text{goal}} - \mathbf{p}_t\|$.
- Positive reward is awarded only when the quadrotor reduces distance to the goal ($+10.0 \cdot \Delta d_t$).
- Collision and ground crash penalties are raised from $-5.0$ to $-50.0$, ensuring that an intentional crash return ($-50.0$) is strictly catastrophic compared to exploratory flight.

---

## 2. Metric Depth Linearization

### Transformation from Z-Buffer to Linear Metric Depth
PyBullet's raw Z-buffer $z_{\text{buf}} \in [0, 1]$ is inverted to true metric distance $z_{\text{linear}}$ via the standard perspective projection formula:
$$z_{\text{linear}} = \frac{z_{\text{far}} \cdot z_{\text{near}}}{z_{\text{far}} - (z_{\text{far}} - z_{\text{near}}) \cdot z_{\text{buf}}}$$
Normalizing linearly by the far plane $z_{\text{far}} = 10.0\text{m}$:
$$\mathbf{I}_{\text{norm}} = \text{clip}\left(\frac{z_{\text{linear}}}{z_{\text{far}}}, 0.0, 1.0\right) \times 255.0$$
This restores full 8-bit dynamic contrast across the entire operational flight range ($0.1\text{m}$ to $10.0\text{m}$), presenting crisp obstacle silhouettes to the CNN.

---

## 3. Egocentric Coordinate Transformation

### Mathematical Body-Frame Rotation
Let $\mathbf{q} = [q_x, q_y, q_z, q_w]^T$ be the quadrotor attitude quaternion. The direction cosine matrix $\mathbf{R}(\mathbf{q}) \in SO(3)$ rotates body-frame vectors into the world frame:
$$\mathbf{v}_{\text{world}} = \mathbf{R} \mathbf{v}_{\text{body}} \implies \mathbf{v}_{\text{body}} = \mathbf{R}^T \mathbf{v}_{\text{world}}$$
Transforming the world displacement $\Delta\mathbf{p}_{\text{world}} = \mathbf{p}_{\text{goal}} - \mathbf{p}_{\text{drone}}$:
$$\mathbf{p}_{\text{target, body}} = \mathbf{R}^T (\mathbf{p}_{\text{goal}} - \mathbf{p}_{\text{drone}})$$
This aligns the goal observation vector with the forward-facing camera orientation, providing full rotation invariance.

---
## Related
#domain/rl-reward #domain/perception #type/investigation #status/resolved
- [[problems/SuicidePolicyExploitation]]
- [[problems/DepthBufferFlattening]]
- [[problems/WorldFrameGoalMisalignment]]
- [[decisions/EgocentricAlignmentAndPotentialReward]]
- [[architecture/RewardFormulation]]
- [[architecture/PerceptionPipeline]]
