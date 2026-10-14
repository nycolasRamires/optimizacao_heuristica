---
date: 2026-09-29
tags: [type/investigation, domain/performance, domain/rl-navigation, status/resolved]
aliases: [VectorizedThroughputAndRetreatBufferInvestigation]
---

> Addresses: [[problems/CPURenderingBottleneck]], [[problems/ObstacleUTrapDeadlock]]
> Resolved by: [[decisions/MultiprocessVectorizationAndRetreatBuffer]]

# Investigation: High-Throughput Parallelization & Multi-Step Retreat Tolerance

## Scope
This investigation addresses the dual challenges of hardware throughput saturation ([[problems/CPURenderingBottleneck]]) and obstacle deadlock in local geometric minima ([[problems/ObstacleUTrapDeadlock]]).

---

## 1. High-Throughput Vectorization Strategy

### A. Multiprocessing Distribution (`SubprocVecEnv`)
Because PyBullet's TinyRenderer is CPU-bound on single-thread software rasterization, distributing the simulation across multiple CPU cores bypasses the Python Global Interpreter Lock (GIL):
- Parallel worker pool: $N_{\text{envs}} = 12$ independent processes.
- Aggregate simulation step time: divided across 12 concurrent workers.

### B. Renderer Optimization (`ER_NO_SEGMENTATION_MASK`)
Passing the flag `flags=p.ER_NO_SEGMENTATION_MASK` instructs PyBullet's C++ rasterizer to completely omit the segmentation buffer allocation, classification, and pixel writing steps, saving $\approx 15\text{--}20\%$ CPU rendering cycles per camera call.

### C. Pipelined CUDA Gradient Updates
Single-step training updates (`train_freq=1`, `gradient_steps=1`) constantly stall the CPU waiting for GPU CUDA kernel completions:
- Setting `train_freq=(4, 'step')` with `gradient_steps=4` collects $4 \times 12 = 48$ transitions before dispatching 4 batched gradient updates.
- Scaling `batch_size = 256` fully utilizes the NVIDIA RTX 5070 tensor cores.
- **Result:** Throughput increased from $80.5\text{ steps/s}$ to **$589.8\text{ steps/s}$** ($7.3\times$ acceleration). Training time dropped from $12.1\text{ hours}$ to $1.65\text{ hours}$.

---

## 2. Formulation of the 3D Multi-Step Retreat Tolerance Buffer

### Mathematical Formulation
To give the quadrotor freedom to retreat around cylindrical obstacles without suffering punitive distance penalties:
1. Maintain a rolling ring buffer of recent goal distances $\mathcal{D}_t = [d_{t-K+1}, \dots, d_t]$ with capacity $K = 10$.
2. Compute recent best distance:
   $$d_{\text{min\_recent}} = \min_{d \in \mathcal{D}_t} d$$
3. Measure distance retreat:
   $$\Delta_{\text{ret}} = d_t - d_{\text{min\_recent}}$$
4. Apply the metric tolerance slack $\delta_{\text{tol}} = 0.10\text{m}$ and retreat multiplier $k_{\text{ret}} = 2.0$:
   $$P_{\text{retreat}}(t) = \begin{cases} k_{\text{ret}} \cdot (\Delta_{\text{ret}} - \delta_{\text{tol}}), & \text{if } \Delta_{\text{ret}} > \delta_{\text{tol}} \\ 0, & \text{otherwise} \end{cases}$$

### Behavioral Impact
The quadrotor is granted up to $10\text{ cm}$ of geometric displacement leeway to reverse or perform wide lateral circumventions around obstacles, completely eliminating U-trap deadlock while preventing unbounded runaway drift.

---
## Related
#domain/performance #domain/rl-navigation #type/investigation #status/resolved
- [[problems/CPURenderingBottleneck]]
- [[problems/ObstacleUTrapDeadlock]]
- [[decisions/MultiprocessVectorizationAndRetreatBuffer]]
- [[architecture/HighThroughputProfiling]]
- [[architecture/RewardFormulation]]
