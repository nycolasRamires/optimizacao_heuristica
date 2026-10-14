---
date: 2026-09-29
tags: [type/architecture, domain/performance, status/active]
aliases: [HighThroughputProfiling, SimulationProfiling]
---

# High-Throughput Optimization & Execution Profiling

## Micro-Profiling Wall-Clock Breakdown
Execution profiling of a single environment step ($25\text{ ms}$ simulated control step, 5 PyBullet sub-steps) on an Intel Core Ultra 9 285K + NVIDIA RTX 5070:

| Pipeline Stage | Wall Clock Time | % of Step Time | Hardware Bottleneck |
| :--- | :--- | :--- | :--- |
| **PyBullet Physics (5 sub-steps)** | $0.049\text{ ms}$ | $1.7\%$ | CPU arithmetic |
| **Contact Manifold Collision Check** | $0.001\text{ ms}$ | $0.03\%$ | PyBullet C++ internal table |
| **Camera Rendering (`ER_TINY_RENDERER`)** | $2.571\text{ ms}$ | $87.7\%$ | CPU single-core rasterization |
| **Kinematics & Python State Logic** | $0.310\text{ ms}$ | $10.6\%$ | NumPy array slicing |
| **Total Environment Step** | **$2.93\text{ ms}$** | $100\%$ | Single CPU thread bound |

When paired with a baseline SAC loop (`train_freq=1`, `batch_size=128`), PyTorch CUDA launch latency and CPU-GPU synchronization added **$11.8\text{ ms}$ per step**, bounding throughput to **$80.5\text{ steps/s}$** ($12\text{ hours}$ for 3.5M steps).

---

## Vectorized Scaling Pipeline
1. **Parallel Worker Multiprocessing (`SubprocVecEnv`):**
   - Distributes $N_{\text{envs}} = 12$ independent PyBullet instances across physical CPU cores.
   - Eliminates Python Global Interpreter Lock (GIL) contention and parallelizes software depth rendering.
2. **PyBullet Rendering Optimization (`ER_NO_SEGMENTATION_MASK`):**
   - Passing `flags=p.ER_NO_SEGMENTATION_MASK` instructs the PyBullet C++ rasterizer to skip segmentation buffer allocation and pixel classification.
3. **Pipelined Gradient Updates:**
   - Setting `train_freq=(4, 'step')` with `gradient_steps=4` aggregates 4 simulation steps ($4 \times 12 = 48$ transitions) before dispatching 4 batched CUDA gradient updates, completely amortizing GPU synchronization stalls.
4. **Batch Size Scaling:**
   - Increased `batch_size=256` to saturate RTX 5070 tensor cores.
5. **Observed Speedup:**
   - Throughput increased from **$80.5\text{ steps/s}$** to **$589.8\text{ steps/s}$** (**$7.3\times$ speedup**). A 3.5M timestep run drops from **$12.1\text{ hours}$** to **$1.65\text{ hours}$**.

---
## Related
#domain/performance #type/architecture
- [[problems/CPURenderingBottleneck]]
- [[investigations/VectorizedThroughputAndRetreatBuffer]]
- [[decisions/MultiprocessVectorizationAndRetreatBuffer]]
- [[architecture/EnvironmentArchitecture]]
