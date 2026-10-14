---
date: 2026-09-29
tags: [type/problem, domain/performance, status/resolved]
aliases: [CPURenderingBottleneckProblem]
---

> Resolved by: [[decisions/MultiprocessVectorizationAndRetreatBuffer]]

# Problem: CPU Depth Rendering Throughput Bottleneck

## Context & Requirement
Deep reinforcement learning algorithms (such as SAC) require hundreds of thousands to millions of transitions to converge. Achieving practical experiment turnaround requires simulation throughput $\ge 500\text{ steps/s}$.

## Symptoms & Micro-Profiling Breakdown
Systematic profiling of a single environment step ($25\text{ ms}$ simulated control step) on an Intel Core Ultra 9 285K + NVIDIA RTX 5070:

| Pipeline Stage | Wall Clock Time | % of Step Time | Hardware Bound |
| :--- | :--- | :--- | :--- |
| **PyBullet Physics (5 sub-steps)** | $0.049\text{ ms}$ | $1.7\%$ | CPU arithmetic |
| **Contact Manifold Collision Check** | $0.001\text{ ms}$ | $0.03\%$ | PyBullet C++ internal |
| **Camera Rendering (`ER_TINY_RENDERER`)** | $2.571\text{ ms}$ | $87.7\%$ | CPU single-core rasterization |
| **Kinematics & Python State Logic** | $0.310\text{ ms}$ | $10.6\%$ | NumPy array slicing |
| **Total Environment Step** | **$2.93\text{ ms}$** | $100\%$ | Single CPU thread |

### The Systemic Bottleneck
- Single-threaded software rasterization in PyBullet's `ER_TINY_RENDERER` consumed $87.7\%$ of environment step time.
- When paired with single-worker SAC (`train_freq=1`, `batch_size=128`), PyTorch CUDA launch latency and GPU-CPU synchronization stalls added $11.8\text{ ms}$ per step.
- Net training speed was severely bounded to **$80.5\text{ steps/s}$**, requiring $>12.1\text{ hours}$ for a standard 3.5M-step training run.

---
## Related
#domain/performance #type/problem #status/resolved
- [[investigations/VectorizedThroughputAndRetreatBuffer]]
- [[decisions/MultiprocessVectorizationAndRetreatBuffer]]
- [[architecture/HighThroughputProfiling]]
