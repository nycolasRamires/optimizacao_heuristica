---
date: 2026-09-29
tags: [type/decision, domain/rl-reward, domain/perception, status/resolved]
aliases: [Decision5, Decision-005, DecisionEgocentricAlignment]
---

> Resolves: [[investigations/SensoryAndRewardAlignment]]
> Related to: [[problems/SuicidePolicyExploitation]], [[problems/DepthBufferFlattening]], [[problems/WorldFrameGoalMisalignment]]

# Decision: Egocentric Multi-Modal Alignment & Potential-Based Reward Formulation

## Context
The initial 3D environment suffered from 3 coupled flaws:
1. Low $-5.0$ collision penalty paired with static $-0.5 \times d$ distance penalties incentivized suicidal crashes to stop negative score bleeding.
2. Non-linear raw Z-buffer compressed $90\%$ of visual depth into white pixels ($230\text{--}255$).
3. Relative target displacement was in world coordinates rather than the drone's body frame.

## Decision
1. **Reward Function Overhaul:** Replace static distance penalties with potential-based progress shaping ($+10.0 \cdot (d_{t-1} - d_t)$) and increase collision/ground penalties to $-50.0$.
2. **Metric Linear Depth:** Normalize true metric linear depth across the full $[0, 255]$ dynamic range for CNN feature extraction.
3. **Egocentric Alignment:** Rotate relative goal displacement into the drone's body frame ($\mathbf{R}^T \Delta\mathbf{p}_{\text{world}}$).

## Rationale
- Completely eliminates the suicide policy loop by penalizing collisions far more severely than exploratory survival.
- Restores high visual contrast, enabling the CNN to extract clear obstacle boundaries.
- Establishes rotational invariance by aligning visual perception and goal direction in the same body coordinate frame.

---
## Related
#domain/rl-reward #domain/perception #type/decision #status/resolved
- [[problems/SuicidePolicyExploitation]]
- [[problems/DepthBufferFlattening]]
- [[problems/WorldFrameGoalMisalignment]]
- [[investigations/SensoryAndRewardAlignment]]
- [[architecture/RewardFormulation]]
- [[architecture/PerceptionPipeline]]
- [[Decisions]]
