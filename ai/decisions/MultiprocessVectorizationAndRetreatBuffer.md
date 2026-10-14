---
date: 2026-09-29
tags: [type/decision, domain/performance, domain/rl-navigation, status/resolved]
aliases: [Decision6, Decision-006, DecisionVectorizationRetreat]
---

> Resolves: [[investigations/VectorizedThroughputAndRetreatBuffer]]
> Related to: [[problems/CPURenderingBottleneck]], [[problems/ObstacleUTrapDeadlock]]

# Decision: Multiprocess Vectorization & 3D Multi-Step Retreat Buffer

## Context
PyBullet software camera rendering limited single-process training throughput to $80.5\text{ steps/s}$ ($12.1\text{ hours}$ for 3.5M steps), and static progress penalties caused deadlock in obstacle U-traps.

## Decision
1. **Parallel Vectorization:** Distribute simulation across 12 CPU workers via `SubprocVecEnv` with pipelined training (`train_freq=(4, 'step')`, `gradient_steps=4`, `batch_size=256`), disabling segmentation mask rendering (`ER_NO_SEGMENTATION_MASK`).
2. **Multi-Step Retreat Buffer:** Implement a 10-step rolling distance history buffer with $0.10\text{m}$ slack ($\delta_{\text{tol}}$) before penalizing retreat ($k_{\text{ret}} = 2.0$).

## Rationale
- **7.3x Speedup:** Increases training throughput from $80.5\text{ steps/s}$ to $589.8\text{ steps/s}$, cutting a 3.5M-step SAC run from $12.1\text{ hours}$ down to $<1.7\text{ hours}$.
- **Flanking Capability:** Grants the quadrotor up to $10\text{ cm}$ of geometric leeway to reverse or take wide lateral arcs around cylindrical pillars, preventing local-minima entrapment.

---
## Related
#domain/performance #domain/rl-navigation #type/decision #status/resolved
- [[problems/CPURenderingBottleneck]]
- [[problems/ObstacleUTrapDeadlock]]
- [[investigations/VectorizedThroughputAndRetreatBuffer]]
- [[architecture/HighThroughputProfiling]]
- [[architecture/RewardFormulation]]
- [[Decisions]]
