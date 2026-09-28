# Project Rules & Agent Guidelines

## 1. Environment & Execution
- **Python Version:** ALWAYS use **Python 3.12** for all command executions and script calls.
  - Active virtual environment: `.venv/bin/python3.12`
  - When running Python scripts or tests, use: `.venv/bin/python3.12 <script.py>` or ensure `PATH=.venv/bin:$PATH`.
- **Shared Lab Machine:** Always maintain environment packages and changes strictly within `.venv/`.

## 2. AI Documentation & Project Tracking (`ai/` Directory)
All agent-maintained documentation and tracking files (besides this `GEMINI.md` rule file) MUST be placed inside the `ai/` directory:

- **[`ai/goals.md`](file:///home/nycolas/m/ai/goals.md) (Goals & Progress):**
  - Keep continuously updated: mark completed items as `- [x]` and add newly identified items as `- [ ]`.
  - **Conciseness:** Do NOT be overly verbose. Keep descriptions short, clear, and actionable.
  - **Hierarchical Indentation:** Divide main goals into sub-goals indented under their parent.
- **[`ai/explanations.md`](file:///home/nycolas/m/ai/explanations.md) (Technical Explanations & Rationale):**
  - When performing technical tasks (such as simulation optimization, algorithm design, or math formulations), document the technical rationale, profiling data, mathematical derivations, and implementation details here.
- **[`ai/decisions.md`](file:///home/nycolas/m/ai/decisions.md) (Engineering Decisions Log):**
  - Record key architectural, algorithmic, and engineering decisions made during development (including context, alternatives considered, and trade-offs).

## 3. Core Project Context (Branch `2d`)
- **Objective:** Bake and solidify the 2D Omnidirectional Drone Simulation for fast RL prototyping and obstacle avoidance.
- **Key Modules:**
  - [`simple_sim.py`](file:///home/nycolas/m/simple_sim.py): Fast analytical 2D kinematic engine, $10\times10\text{m}$ arena, and 15-ray LiDAR raycaster ($100^\circ$ FOV).
  - [`rl_training.py`](file:///home/nycolas/m/rl_training.py): Gymnasium wrapper ([`OmniDroneEnv`](file:///home/nycolas/m/rl_training.py#L20-L143)), 17D observation space, 3D action space, parameterizable rewards.
  - [`train.py`](file:///home/nycolas/m/train.py): Stable-Baselines3 SAC training loop with vectorized environments.
  - [`enjoy.py`](file:///home/nycolas/m/enjoy.py): Pygame inference visualizer.
  - [`Render.py`](file:///home/nycolas/m/Render.py): Pygame manual keyboard control testbed.
  - [`ai/`](file:///home/nycolas/m/ai): Centralized directory for goals, technical explanations, and decisions.
- **Active Challenges:**
  - GA reward/penalty parameter optimization.
  - Resolving the "U-Trap Paradox" (local minima in feed-forward policies).
  - Preparation for 3D Quadrotor Physics and 6-DOF simulation.

## 4. Long-Running Simulation & GPU Safeguards
- **Zero-Modification Invariant:** NEVER modify, edit, or delete codebase files or scripts while an active background training/evolution process is running on the GPU.
- **Process Verification:** Before editing core files, verify active training status (`ps aux | grep evolution.py`).
- **Deterministic Evaluation Standard:** In SAC and actor-critic setups, always benchmark and rank individuals using deterministic mode (`deterministic=True`) over $\ge 100$ seeded episodes to eliminate evaluation variance.

## 5. Evolutionary Algorithm & Optimizer Rules
- **PyTorch/SB3 Optimizer Parameter Group Override:** When fine-tuning or warm-starting models at modified learning rates, always explicitly update all `param_groups['lr']` in actor, critic, and entropy optimizers (`set_learning_rate`).
- **Tiered Reproduction Separation:**
  - *Group 2 (Elites):* Cloned directly without retraining.
  - *Group 3 (Elite Crossover):* Strictly Elite $\times$ Elite with arithmetic gene blending.
  - *Group 4 (Non-Elite Selection):* Tournament selection strictly among Non-Elite $\times$ Non-Elite candidates to prevent unguided elite dilution.
  - *Group 1 (Replenishment):* Replenishing dead individuals failing viability gates.
- **Mutation Integrity:** All non-elite offspring must pass through the mutation operator ($p_{\text{mut}} > 0$) to preserve diversity and prevent frozen cloning.

## 6. 3D Quadrotor Transition Principles
- **6-DOF Rigid-Body Dynamics:** Account for gravity compensation ($mg$), rotor thrust equations, gyroscopic torque, and aerodynamic drag.
- **Full Markov State Observability:** Include position error $\mathbf{p}$, linear velocity $\mathbf{v}$, attitude quaternion $\mathbf{q} \in \mathbb{S}^3$, and body angular rate $\boldsymbol{\omega}$.
- **3D Spatial Safety:** Use 3D Oriented Bounding Boxes (OBB) or Signed Distance Fields (SDF) to prevent spawn embedding and calculate proximity penalties.
