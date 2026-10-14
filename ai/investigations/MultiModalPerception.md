---
date: 2026-09-29
tags: [type/investigation, domain/perception, status/resolved]
aliases: [MultiModalPerceptionInvestigation]
---

> Addresses: [[problems/Unmapped3DObstaclePerception]]
> Resolved by: [[decisions/MultiModalPerception]]

# Investigation: Multi-Modal Sensor Fusion for 3D Quadrotors

## Sensor Modality Approaches

### 1. Vectorized 3D Spherical LiDAR Rays
- **Representation:** 1D array of radial range distances across elevation and azimuth angles.
- **Pros:** Fast to compute; small observation dimension.
- **Cons:** Sparse spatial coverage; requires custom raycasting loops in PyBullet which can degrade simulation step rates if ray count is high.

### 2. Monocular RGB Camera
- **Representation:** $3 \times H \times W$ color image.
- **Pros:** High realism.
- **Cons:** Computationally heavy; sensitive to lighting; requires depth estimation from monocular cues, which is notoriously data-hungry and prone to scale drift in RL.

### 3. Forward Depth Image + Kinematic State Vector
- **Representation:** Multi-modal dictionary:
  - `"image"`: $1 \times 64 \times 64$ normalized metric depth map rendered along the drone's forward body axis.
  - `"state"`: 23-dimensional continuous kinematic proprioceptive vector.
- **Network Architecture:** SB3 `MultiInputPolicy` with a composite feature extractor:
  - A 2D Convolutional Neural Network (CNN) extracts spatial obstacle features from `"image"`.
  - A Multi-Layer Perceptron (MLP) extracts dynamic motion features from `"state"`.
  - The feature embeddings are concatenated into a unified latent representation for the SAC actor and critic networks.

## Outcome
The dictionary observation structure provides the optimal balance of rich spatial geometry and precise kinematic state feedback.

---
## Related
#domain/perception #type/investigation #status/resolved
- [[problems/Unmapped3DObstaclePerception]]
- [[decisions/MultiModalPerception]]
- [[architecture/PerceptionPipeline]]
