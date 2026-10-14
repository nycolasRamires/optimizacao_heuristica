---
date: 2026-09-29
tags: [type/architecture, domain/perception, status/active]
aliases: [PerceptionPipeline, VisionPipeline]
---

# Multi-Modal Perception & Observation Space Pipeline

The policy consumes a multi-modal dictionary observation space:
$$\mathcal{O} = \{\text{"image"}: \mathbf{I} \in \mathbb{R}^{1 \times 64 \times 64}, \; \text{"state"}: \mathbf{s} \in \mathbb{R}^{23}\}$$

---

## 1. 23D Kinematic Proprioception Vector ($\mathbf{s}$)
$$\mathbf{s} = [\mathbf{p}, \mathbf{q}, \boldsymbol{\eta}, \mathbf{v}_{\text{clipped}}, \boldsymbol{\omega}_{\text{clipped}}, \boldsymbol{\Omega}_{\text{motor}}, \mathbf{p}_{\text{target, body}}]^T \in \mathbb{R}^{23}$$

1. $\mathbf{p} = [x, y, z]^T \in \mathbb{R}^3$: World position coordinate.
2. $\mathbf{q} = [q_x, q_y, q_z, q_w]^T \in \mathbb{S}^3$: Attitude quaternion.
3. $\boldsymbol{\eta} = [\phi, \theta, \psi]^T \in \mathbb{R}^3$: Roll, pitch, and yaw Euler angles.
4. $\mathbf{v}_{\text{clipped}} \in \mathbb{R}^3$: Linear velocity vector, clipped to $[-10, 10]\text{ m/s}$.
5. $\boldsymbol{\omega}_{\text{clipped}} \in \mathbb{R}^3$: Body angular velocity vector, clipped to $[-20, 20]\text{ rad/s}$.
6. $\boldsymbol{\Omega}_{\text{motor}} \in \mathbb{R}^4$: Current 4 motor RPMs.
7. $\mathbf{p}_{\text{target, body}} = \mathbf{R}(\mathbf{q})^T (\mathbf{p}_{\text{goal}} - \mathbf{p}) \in \mathbb{R}^3$: Relative 3D displacement vector to target rotated into the **drone's egocentric body frame**.

---

## 2. Forward Depth Vision Pipeline ($\mathbf{I}$)
Rendered at the drone's position $\mathbf{p}$ pointing along the body forward axis:
$$\mathbf{d}_{\text{forward}} = \mathbf{R}(\mathbf{q}) \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{u}_{\text{up}} = \mathbf{R}(\mathbf{q}) \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

- **Camera Specs:** Resolution $64 \times 64$, Field of View $\text{FOV} = 90^\circ$, near plane $z_{\text{near}} = 0.1\text{m}$, far plane $z_{\text{far}} = 10.0\text{m}$.
- **View Matrix:** Computed via:
  ```python
  p.computeViewMatrix(
      cameraEyePosition=pos,
      cameraTargetPosition=pos + forward_vec,
      cameraUpVector=up_vec
  )
  ```
- **Depth Linearization & Normalization:**
  PyBullet's raw Z-buffer $z_{\text{buf}}$ is converted to metric distance $z_{\text{linear}}$:
  $$z_{\text{linear}} = \frac{z_{\text{far}} \cdot z_{\text{near}}}{z_{\text{far}} - (z_{\text{far}} - z_{\text{near}}) \cdot z_{\text{buf}}}$$
  $$\mathbf{I} = \text{clip}\left(\frac{z_{\text{linear}}}{z_{\text{far}}}, 0.0, 1.0\right) \times 255.0$$
  This yields a normalized 8-bit array $[1, 64, 64]$ with maximum dynamic contrast between obstacles and open space.

---
## Related
#domain/perception #type/architecture
- [[decisions/MultiModalPerception]]
- [[decisions/EgocentricAlignmentAndPotentialReward]]
- [[problems/DepthBufferFlattening]]
- [[problems/WorldFrameGoalMisalignment]]
- [[architecture/EnvironmentArchitecture]]
