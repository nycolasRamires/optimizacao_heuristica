---
date: 2026-09-29
tags: [type/problem, domain/perception, status/resolved]
aliases: [DepthBufferFlatteningProblem]
---

> Resolved by: [[decisions/EgocentricAlignmentAndPotentialReward]]

# Problem: Non-Linear Depth Buffer Flattening & Contrast Loss

## Context & Requirement
The convolutional neural network (CNN) in the vision head of `MultiInputPolicy` requires sufficient contrast and dynamic range in the forward depth map to differentiate obstacle edges, distances, and free space.

## Symptoms & Mathematical Diagnosis
The environment previously computed the observation image via:
```python
image_obs = (depth_array * 255.0).astype(np.uint8).transpose(2, 0, 1)
```
Where `depth_array` returned by PyBullet's `getCameraImage` is the standard OpenGL non-linear Z-buffer value:
$$z_{\text{buffer}} = \frac{z_{\text{far}} (z - z_{\text{near}})}{z (z_{\text{far}} - z_{\text{near}})}$$

For camera near plane $z_{\text{near}} = 0.1\text{m}$ and far plane $z_{\text{far}} = 10.0\text{m}$:
- At $z = 1.0\text{m}$: $z_{\text{buffer}} = 0.909$ ($\times 255 = 231$)
- At $z = 3.0\text{m}$: $z_{\text{buffer}} = 0.973$ ($\times 255 = 248$)
- At $z = 5.0\text{m}$: $z_{\text{buffer}} = 0.986$ ($\times 255 = 251$)

### Consequence
Over $90\%$ of all operational flight distances were compressed into an extremely narrow pixel intensity band of $[230, 255]$. The depth image was visually washed out, almost entirely white, obliterating visual spatial gradients and preventing the CNN from detecting obstacles until physical impact.

---
## Related
#domain/perception #type/problem #status/resolved
- [[investigations/SensoryAndRewardAlignment]]
- [[decisions/EgocentricAlignmentAndPotentialReward]]
- [[architecture/PerceptionPipeline]]
