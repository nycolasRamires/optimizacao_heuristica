# Project Engineering Decisions (`ai/decisions.md`)

This file logs architectural, algorithmic, and engineering decisions made during development, recording context, alternatives considered, and trade-offs.

---

## Decision 1: Pivot to 2D Analytical Simulation on `2d` Branch
- **Date:** 2026-09-08
- **Context:** The `main` branch utilized 3D quadcopter physics (`gym-pybullet-drones`) with depth cameras and PID control, resulting in slow training iterations and high simulation overhead.
- **Decision:** Establish the `2d` branch with an analytical 2D kinematic engine ([`simple_sim.py`](file:///home/nycolas/m/simple_sim.py)) and planar 15-ray LiDAR raycaster.
- **Rationale:** Decouples navigation algorithms, obstacle avoidance heuristics, and RL training from 3D aerodynamic complexity, allowing orders-of-magnitude faster iteration.

---

## Decision 2: Pure NumPy Vectorization over Compiled Extensions (Numba/Cython)
- **Date:** 2026-09-19
- **Context:** Simulation throughput needed a $>4\text{x}$ boost to hit the $<30\text{ min} / 1\text{M steps}$ target. The system is a shared laboratory machine.
- **Alternatives Considered:**
  1. *Numba JIT (`@njit`):* Requires adding compilation toolchains and external dependencies.
  2. *C++ / Cython extension:* Increases build complexity and platform-dependent artifacts.
  3. *Pure NumPy array broadcasting:* Uses native vectorization already linked against high-performance BLAS.
- **Decision:** Vectorized raycasting and wall projections using pure NumPy broadcasting.
- **Rationale:** Keeps dependencies clean, fully contained within `.venv`, zero compilation step, and achieved 40,750 FPS—far exceeding the throughput requirements.

---

## Decision 3: In-Process `DummyVecEnv` Parallelization (16 Envs)
- **Date:** 2026-09-19
- **Context:** Single-environment SAC was throttled by GPU CUDA launch latency (167 steps/s, ~100 min for 1M steps).
- **Alternatives Considered:**
  1. *`SubprocVecEnv` (multiprocessing):* Spawns separate Python processes. For complex physics engines this helps, but for ultra-fast analytical environments (40,000 FPS), inter-process IPC, serialization, and pipe latency introduce significant slowdowns.
  2. *`DummyVecEnv` (in-process):* Steps $N$ environments sequentially in memory, then batches all observations together for the neural network.
- **Decision:** Use `DummyVecEnv` with 16 parallel environments for training in [`train.py`](file:///home/nycolas/m/train.py).
- **Rationale:** Reached **2,584.8 steps/s** (~6.45 minutes for 1M steps), running completely in-process without multiprocessing synchronization bottlenecks.

---

## Decision 4: Parameterized Reward Interface for Genetic Algorithm Optimization
- **Date:** 2026-09-19
- **Context:** The next project milestone is implementing a Genetic Algorithm (GA) to automatically search and optimize reward and penalty hyperparameters.
- **Decision:** Add `DEFAULT_REWARD_PARAMS` dictionary and a `reward_params=None` parameter to [`OmniDroneEnv.__init__`](file:///home/nycolas/m/rl_training.py#L27-L45).
- **Rationale:** Allows GA optimization scripts to pass candidate chromosomes directly to environment instances without monkey-patching, global variables, or subclassing.

---

## Decision 5: Centralized AI Tracking Files under `ai/` Directory
- **Date:** 2026-09-19
- **Context:** Project requires systematic tracking of progress, technical explanations, and design decisions while preserving a clean repository root.
- **Decision:** Group agent documentation files inside `ai/`:
  - `ai/goals.md`: Project goals and progress checklist.
  - `ai/explanations.md`: In-depth technical rationale and mathematical explanations.
  - `ai/decisions.md`: Engineering decision log.
  - Root `GEMINI.md`: Direct agent workspace guidelines file referencing `ai/`.
- **Rationale:** Separates code from AI metadata and maintains clear project governance.

---

## Decision 6: Pre-Training `base` Policy with Warm-Start Learning Rate Adaptation for GA
- **Date:** 2026-09-19
- **Context:** Training each Genetic Algorithm individual from tabula rasa (random initialization) over hundreds of thousands of steps wastes compute on relearning basic locomotion.
- **Alternatives Considered:**
  1. *Train from scratch for every individual:* Clean, but requires $>50\text{M}$ steps across generations.
  2. *Freeze backbone / linear probing:* Fast, but restricts the policy from modifying non-linear collision avoidance maneuvers.
  3. *Warm-start with elevated learning rate:* Pre-train a 500k-step `base.zip` policy, then fine-tune each candidate for shorter horizons while boosting the learning rate (e.g. $1\times 10^{-3}$) so the candidate's custom reward shaping quickly overrides prior policy bias.
- **Decision:** Adopt warm-start fine-tuning from `base.zip` with explicit PyTorch optimizer parameter group learning rate updates (`set_learning_rate` in [`train.py`](file:///home/nycolas/m/train.py)).
- **Rationale:** Cuts computational cost per individual by $>80\%$ while allowing rapid behavioral divergence driven by candidate reward functions.

---

## Decision 7: Expanded 20m x 20m Arena with Dual Asymmetric Pillars & Dual-Layer Spawn Verification
- **Date:** 2026-09-19
- **Context:** The previous 10m x 10m arena with a single barrier was too small and simple, failing to test multi-obstacle avoidance and dynamic path navigation across arbitrary spawn orientations.
- **Alternatives Considered:**
  1. *Procedural random mazes on every reset:* Too much environmental non-stationarity for early SAC training.
  2. *Single larger barrier:* Does not test multi-obstacle bypass or navigation between adjacent obstacles.
  3. *Static 20m x 20m arena with two asymmetric pillars:* Large Pillar ($4\text{m} \times 4\text{m}$) and Small Pillar ($2.5\text{m} \times 2.5\text{m}$), leaving $3\text{m}$ to $11\text{m}$ wide corridors throughout the arena.
- **Decision:** Adopt the 20m x 20m arena with two asymmetric square pillars. Combine line-segment wall distance checking with bounding-box Minkowski expansion (`_is_inside_any_pillar`) to strictly prevent any entity from spawning or moving inside pillar interiors.
- **Rationale:** Expands the state space ($4\times$ area), guarantees 100% topological accessibility between any spawn point and goal, and forces the policy to generalize navigation across varied headings and obstacle profiles.

---

## Decision 8: Rescaling Arena to 12m x 12m with High-Obstacle Density Corridors
- **Date:** 2026-09-19
- **Context:** The 20m x 20m arena proved too easy, providing overly wide corridors ($5\text{m}$ to $11\text{m}$) where the drone was rarely constrained by obstacles.
- **Alternatives Considered:**
  1. *Add more pillars to 20m arena:* Increases wall segment count and geometric complexity.
  2. *Rescale arena to 12m x 12m:* Maintains the clean two-pillar setup while compressing corridors down to $1.8\text{m}$–$3.2\text{m}$.
- **Decision:** Rescale the arena to $12\text{m} \times 12\text{m}$ with a $2.8\text{m} \times 2.8\text{m}$ large pillar and a $1.8\text{m} \times 1.8\text{m}$ small pillar.
- **Rationale:** Dramatically increases challenge: the drone must navigate corridors between $1.8\text{m}$ and $3.2\text{m}$ wide (only $4.5\times$ drone diameter), forcing active LiDAR-based steering and fine obstacle avoidance without creating impassable bottlenecks.

---

## Decision 9: Compact 10m x 10m Arena with 3 Asymmetric Pillars for Elevated Navigation Difficulty
- **Date:** 2026-09-19
- **Context:** Even in the 12m arena, the policy achieved high navigation success rapidly. To prepare an adequately challenging environment for the upcoming Genetic Algorithm reward tuning, the arena was tightened and densified.
- **Alternatives Considered:**
  1. *Add dead-end walls:* Risks creating insurmountable local minima for memoryless feed-forward policies before introducing recurrent/hybrid planners.
  2. *Add 1 additional small pillar in a 10m x 10m arena:* Distributes 3 asymmetric obstacles across quadrants, compressing free space while maintaining open convex corridors.
- **Decision:** Establish a 10m x 10m arena with 3 pillars (1 Large: $2.2\text{m} \times 2.2\text{m}$, 2 Small: $1.5\text{m} \times 1.5\text{m}$ and $1.2\text{m} \times 1.2\text{m}$).
- **Rationale:** Obstacles frequently occlude direct line-of-sight between randomized start poses and goals, forcing the policy to execute non-trivial multi-turn avoidance maneuvers while preserving topological connectivity across all corridors ($1.8\text{m}$–$3.1\text{m}$).

---

## Decision 10: Downscale Baseline Pre-Training to 300,000 Steps to Preserve Genetic Plasticity
- **Date:** 2026-09-19
- **Context:** A 500k-step baseline achieved ~75%–80% goal convergence with deeply crystallized value functions and low action entropy, making subsequent fine-tuning resistant to reward/cost gradient shifts during GA exploration.
- **Alternatives Considered:**
  1. *500k-step baseline:* Near-converged policy; leaves little gradient room for GA reward variations to steer navigation behaviors.
  2. *100k-step baseline:* Policy is still mostly random wandering and fails to reach distant goals, requiring lengthy fine-tuning per GA candidate.
  3. *300k-step baseline:* Balances basic locomotion and obstacle awareness (64% success, 29% collision, 7% timeout) while preserving high policy entropy and adaptability.
- **Decision:** Train the baseline model (`base.zip`) for exactly 300,000 timesteps (~2 minutes across 16 vectorized environments).
- **Rationale:** The 300k checkpoint has learned basic velocity control, LiDAR perception, and coarse goal tracking, but retains enough behavioral variance and plasticity for GA-evolved reward functions to meaningfully differentiate fitness and reshape flight policies.

---

## Decision 11: Multi-Step Retreat Penalty with Tolerance & Timestep-Decayed Early Goal Bonus
- **Date:** 2026-09-19
- **Context:** Standard progress shaping penalizes any microscopic backward step, which can trap drones when circumnavigating convex obstacles. Furthermore, constant goal reward provides no positive incentive to finish in 50 steps rather than 490 steps beyond passive living penalties.
- **Alternatives Considered:**
  1. *Immediate single-step retreat penalty without tolerance:* Overly punitive; drones navigating around pillar corners must sometimes increase Euclidean distance temporarily to bypass obstacles.
  2. *Pure episode-length reward replacing fixed goal reward:* Replaces sparse success signal and disrupts return scale.
  3. *Rolling history window with distance tolerance + additive linear early arrival bonus:* Maintains a rolling $N$-step deque (`retreat_window_steps: 10`) of closest approach to the goal. Drones are only penalized if $d_t - d_{\min, \text{recent}} > \delta_{\text{tol}}$. In addition, a linear bonus $k_{\text{early}} \cdot (\text{max\_steps} - t)$ is added on top of the fixed `goal_reward`.
- **Decision:** Implement both policies into `OmniDroneEnv` and `DEFAULT_REWARD_PARAMS`.
- **Rationale:** Preserves necessary local maneuvering around obstacles while curbing aimless wandering, and creates a strong gradient for rapid, direct trajectories without degrading the primary success incentive.

---

## Decision 12: Transition to 2nd-Order Acceleration Control & Speed-Regulating Policies
- **Date:** 2026-09-19
- **Context:** Direct velocity control ($\dot{\mathbf{p}} = \mathbf{v}$) abstracts away inertial dynamics, allowing instantaneous directional switching without deceleration. Real quadrotors actuate thrust to induce accelerations ($\ddot{\mathbf{p}} = \mathbf{a}$), requiring active braking and speed management.
- **Alternatives Considered:**
  1. *Maintain 1st-order velocity kinematics:* Simpler RL convergence, but unrealistic flight dynamics; policies learn non-physical zero-inertia cornering.
  2. *Full 3D aerodynamic simulation:* Too computationally heavy for genetic reward hyperparameter search.
  3. *2nd-order planar acceleration kinematics with physical velocity saturation and 19D observations:* Commands $[a_x, a_y]$ within $[-3.0, 3.0]\text{m/s}^2$ while preserving direct yaw rate control ($\omega$). Physical speed is hard-clamped at $v_{\max} = 2.0\text{m/s}$. Adds local velocities $(v_x, v_y)$ to observation (19D) to satisfy full state observability (Markov property).
- **Decision:** Implement 2nd-order acceleration kinematics in [`simple_sim.py`](file:///home/nycolas/m/simple_sim.py) and [`rl_training.py`](file:///home/nycolas/m/rl_training.py), along with 4 speed-dependent reward policies:
  1. *High Speed Reward:* $k_{\text{spd}} \cdot (v / v_{\max})$
  2. *Speed-Dependent Crash Penalty:* Base collision penalty $+ k_{\text{crash}} \cdot (v_{\text{impact}} / v_{\max}) + k_{\text{high\_crash}} \cdot \text{excess}$ when $v_{\text{impact}} > v_{\text{safe}}$.
  3. *Overspeed Flight Penalty:* $k_{\text{over}} \cdot ((v - v_{\text{safe}}) / (v_{\max} - v_{\text{safe}}))$ for exceeding fixed $v_{\text{safe}} = 1.2\text{m/s}$.
  4. *Constant Velocity Cruise Reward:* Rewards low inter-step speed fluctuation $\Delta v \approx 0$ during motion ($v > 0.2\text{m/s}$).
- **Rationale:** Introduces physical realism, demands deliberate deceleration before corners, and establishes a balanced multi-objective trade-off between speed and collision safety for GA optimization.

---

## Decision 13: Genetic Algorithm Architecture, Concurrency, and Modular Operator Stubs
- **Date:** 2026-09-19
- **Context:** Training 10 individuals per generation across 20 generations for 500k timesteps each equals 100M total simulation steps. Concurrency and clean operator separation are critical.
- **Alternatives Considered:**
  1. *Sequential individual training:* 1 worker at a time under-utilizes the RTX 5070 GPU and 24-core CPU.
  2. *Heavy ray / dask distributed cluster:* High overhead and external dependency footprint for a single lab workstation.
  3. *In-process multiprocessing with 'spawn' context and dual-worker concurrency:* 2 parallel SAC workers running simultaneously on CUDA. Each worker manages an 8-environment `DummyVecEnv` (16 envs total across workers).
- **Decision:** Build [`evolution.py`](file:///home/nycolas/m/evolution.py) with:
  - 2-worker concurrent training pool using `multiprocessing.get_context('spawn')`.
  - Checkpoint caching to allow graceful interruptions and resumability.
  - Distinct modular stubs (`select_elites`, `recombine`, `mutate`) marked with `pass` to allow plug-and-play definition of genetic policies.
- **Rationale:** Maximize hardware utilization without memory thrashing, maintain clean code isolation, and provide an extensible testbed for user-specified evolutionary policies.

---

## Decision 14: Viability Threshold Survival Filter (50% SR), Timesteps-to-Goal Fitness, and 20-Individual Population
- **Date:** 2026-09-19
- **Context:** An aggregate reward or composite scalar fitness score can obscure whether an agent actually reached the objective efficiently versus merely accumulating proximity/speed shaping points. Furthermore, a 10-individual population risks premature genetic drift in a 19-dimensional search space.
- **Alternatives Considered:**
  1. *Composite scalar return:* Conflates reward weights (which differ for each individual) with actual physical performance.
  2. *Pure success rate ranking:* Leads to ties among top performers once multiple individuals reach similar high success rates.
  3. *Two-stage viability filtering with physical navigation efficiency ranking:*
     - Viability Gate: An individual must achieve $\text{success\_rate} \ge 0.50$ (50%) over 100 test tries to be deemed viable. If $< 50\%$, it dies immediately and is replaced by a random immigrant.
     - Fitness Ranking: Only viable individuals are ranked, sorted strictly by average timesteps taken to reach the goal across successful episodes (fewer timesteps = higher rank).
- **Decision:** Adopt the viability gate ($\text{SR} \ge 50\%$) with average-timesteps-to-goal fitness evaluated over 100 seeded tries, and expand population size to 20 individuals per generation.
- **Rationale:** Strictly eliminates unfit reward genomes that cannot achieve basic navigational viability, while providing an objective physical benchmark (time to goal) that is completely independent of the candidate's internal reward scaling.

---

## Decision 15: Multi-Group Evolutionary Reproduction Cycle with 3-Gene Blend Crossover and Rank Selection
- **Date:** 2026-09-20
- **Context:** The evolutionary transition from generation $g$ to $g+1$ requires balancing strict elitism, genetic recombination of high-performing lineages, probabilistic survival of diverse viable solutions, and replenishment of non-viable individuals ($SR < 0.50$).
- **Alternatives Considered:**
  1. *Standard Roulette Wheel / Tournament Selection:* May cause rapid convergence to a single lineage or lose proven elite reward sets without multi-group structural guarantees.
  2. *Uniform crossover across all genes vs. Arithmetic Blend:* Discrete crossover cannot create intermediate parameter values; full blend across all 17 genes risks washing out distinct behavioral traits into homogenous averages.
  3. *Multi-Group Allocation with Partial 3-Gene Blending:*
     - **Group 2 ($n_{\text{best}} = 4$):** 4 best surviving individuals pass directly as unmutated clones.
     - **Group 3 ($C(4, 2) = 6$):** All $\binom{4}{2} = 6$ pairs of elites are recombined producing 12 children (2 per pair); a random half ($6$) is selected and mutated.
     - **Group 4 ($n_{\text{left}} = N - n_{\text{dead}} - n_{\text{best}} - C(n_{\text{best}}, 2)$):** Non-elite survivors are selected without replacement proportional to rank weight $W_r = (n_{\text{survivors}} - r + 1)$, passed as unmutated clones.
     - **Group 1 ($n_{\text{dead}}$):** Offspring generated by completely random recombination of survivor pairs, mutated.
     - **Crossover Operator:** Exactly 3 randomly chosen genes are blended via arithmetic crossover ($c_1 = \alpha p_1 + (1-\alpha) p_2, c_2 = (1-\alpha) p_1 + \alpha p_2$), while the remaining 14 genes undergo uniform discrete crossover.
     - **Mutation Operator:** Gaussian perturbation $\Delta \sim \mathcal{N}(0, (0.1(high-low))^2)$ with fixed rate $p_{\text{mut}} = 0.1$, clamped to bounds.
     - **High-Mortality Fallback:** If $n_{\text{dead}} > 10$ ($n_{\text{survivors}} < 10$), prioritize survivors (up to 4 elites + up to 6 elite offspring) and fill remaining slots up to 20 with random recombination offspring (or random immigrants if $< 2$ survive).
- **Decision:** Implement the Multi-Group Allocation with 3-Gene Blend Crossover, rank-weighted non-elite selection without replacement, and high-mortality survivor prioritization.
- **Rationale:** Preserves top verified reward weights intact (elitism), explores fine-grained continuous variations among top lineages (elite blend crossover), preserves genetic diversity from lower-ranked survivors, and ensures mathematically guaranteed population size of 20 under any mortality rate.

---

## Decision 16: Multi-Perspective Evolutionary Visual Analytics Suite (`plot_evolution.py`)
- **Date:** 2026-09-21
- **Context:** To monitor convergence, evaluate selection dynamics, and debug evolutionary trends, the user requested three specific visual analyses:
  1. Individual existence/persistence across all generations.
  2. Best individual performance metrics over time.
  3. Population mean metrics and spread over time.
- **Alternatives Considered:**
  1. *Scalar logging only (text/JSON):* Difficult to track genealogies or detect whether high performance is driven by a single dominant lineage versus diverse convergent families.
  2. *Single generic convergence plot:* Conflates elite progress with population mean and obscures mortality and lineage survival.
  3. *Tri-Plot Architecture with Automatic Generation Hooks:*
     - **Plot 1 (Individual Existence & Lineage Tracker):** Identifies unique genomes by parameter signatures, plots horizontal timeline ribbons spanning from birth to death with role markers (`elite`, `rank_selected`, `recomb`, `init`), and displays population origin composition and lifespan histograms.
     - **Plot 2 (Best Individual Metrics):** Tracks the generation champion's timesteps to goal, success/collision/timeout rates, flight speed, and return.
     - **Plot 3 (Population Mean & Spread):** Tracks the population average with $\pm 1\sigma$ envelopes, min/max bounds, and the selective advantage gap ($\Delta$ steps between best and mean).
     - Hooked directly into `evolution.py` to auto-refresh at the end of every generation.
- **Decision:** Implement [`plot_evolution.py`](file:///home/nycolas/m/plot_evolution.py) and embed real-time plot generation into [`evolution.py`](file:///home/nycolas/m/evolution.py).
- **Rationale:** Gives immediate visual feedback on evolutionary dynamics, verifies elite preservation and recombination efficacy, and facilitates diagnosis of convergence rates in high-dimensional continuous reward spaces.

---

## Decision 17: Evolutionary Upgrades: 20-Individual Matrix, 17-Gene Evolution Plot, 55% Viability Threshold, Tournament Selection, and Bounded ±50% Mutation
- **Date:** 2026-09-21
- **Context:** Following the 5-generation pilot run, the user requested greater visual clarity regarding individual persistence (ensuring all 20 individuals per generation are clearly accounted for), explicit visual markers for mutation events, enumeration of dead candidates, a comprehensive 17-gene evolution plot, increased selective pressure, reduced mutation volatility, dual champion reporting, and generation-to-generation progression tracking.
- **Alternatives Considered:**
  1. *Maintain unique genome lifespan timeline for Plot 1A:* Confusing because rows represented unique genome lifespans across the entire run rather than individual slots in each generation column.
  2. *Retain rank-proportional survivor selection:* Slower selective pressure, occasionally permitting lower-performing individuals to out-survive higher-ranked non-elites.
  3. *Retain large Gaussian mutation noise:* Perturbations based on overall search domain range ($0.10 \times \text{range}$) caused erratic parameter jumps that broke finely-tuned policies.
- **Decision:**
  1. **Plot 1A Redesign:** Fixed 20-slot matrix ($y = \text{Rank 1}$ to $\text{Rank 20}$) in each generation column ($x = \text{Gen 0}$ to $\text{Gen 4}$). Direct clones connected with solid lines; mutated/recombined offspring connected with dashed lines and prominent **⚡ mutation badges** at transition midpoints. Failed individuals ($< 55\%$ SR) explicitly plotted at bottom slots with bold red `[DEAD]` markers and failure stats.
  2. **Plot 1B & Enumeration:** Fixed-height stacked bars of 20 with red hatched segments for dead individuals, accompanied by an explicit mortality enumeration callout card.
  3. **Plot 4 (17-Gene Evolution Grid):** 6x3 subplot matrix tracking population mean, $\pm 1\sigma$ envelope, best individual trajectory, search space boundaries, individual candidate alleles (N=20), and an evolution summary card detailing top adaptive increases, decreases, and most converged genes.
  4. **Viability Threshold:** Increased from $0.50 \to 0.55$ (55% SR over 100 evaluation episodes).
  5. **Tournament Selection:** Implemented $k = 3$ tournament selection without replacement for Group 4 survivors.
  6. **Bounded $\pm 50\%$ Mutation:** Mutation replaced with relative $\pm 50\%$ perturbation ($v \pm 0.5|v|$) clamped to `GENE_BOUNDS`.
  7. **Dual Champion Reporting:** Output both the final generation best policy and the global all-time champion across all generations.
  8. **Progression Delta Tracking:** Display two-generation comparison deltas ($\Delta\text{Steps}$, $\Delta\text{SR}$, $\Delta\text{Coll}$).
---

## Decision 18: High-Throughput Training Acceleration (Numba Raycaster, SAC TrainFreq Batching, and Compiled Policy)
- **Date:** 2026-09-21
- **Context:** Scaling the genetic algorithm to 30 individuals across 30 generations (900 individuals $\times$ 300k steps = 270M total steps) required maximizing single-node throughput on the RTX 5070 and multi-core CPU.
- **Alternatives Considered:**
  1. *Leave SAC train_freq=1 with per-step updates:* High PyTorch CUDA kernel launch overhead causes CPU micro-stalls alternating between 1 environment step and 1 gradient update.
  2. *Asynchronous distributed workers (Ray/Dask):* High orchestration complexity and memory duplication on a single workstation.
  3. *Three-Tier Optimization Stack:*
     - **Optimization 4 (Numba JIT Engine):** Compile `read_sensors()` and `_get_min_dist_to_walls()` using `@njit(fastmath=True)` with automatic NumPy fallback.
     - **Optimization 1 (SAC Batch Pipelining):** Configure `train_freq=(4, 'step')` and `gradient_steps=4`. Preserves exact 1:1 Update-to-Data (UTD) ratio while pipelining 4 consecutive GPU forward/backward updates.
     - **Optimization 3 (PyTorch Compilation):** Compile `model.policy.actor` via `torch.compile`, paired with automatic unwrapping (`_orig_mod`) prior to `model.save()` to prevent checkpoint serialization errors.
- **Decision:** Implement the Three-Tier Optimization Stack across `simple_sim.py`, `train.py`, and `evolution.py`.
- **Empirical Results:**
  - Raycaster call latency dropped from 155.6 ms to 5.04 ms per 10k calls (**30.8x faster**).
  - Wall distance latency dropped from 56.7 ms to 1.86 ms per 10k calls (**30.5x faster**).
  - Training throughput increased from **2,450.2 steps/s** (baseline, 16 envs) $\to$ **3,025.6 steps/s** (+23.5% with 16 envs) $\to$ **3,801.5 steps/s** (+55.1% with 20 envs).
- **Rationale:** Reduces full 30-individual generation training time from ~50 minutes down to ~20–25 minutes without sacrificing sample efficiency or model checkpoint compatibility.

---

## Decision 19: Standalone Evolutionary Visualizations, Dynamic Population Scaling, In-Symbol Mutation Badges, and Embedded HTML Dashboard
- **Date:** 2026-09-21
- **Context:** Previously, Plot 1 packed 1A (matrix), 1B (composition), and 1C (dispersion) into a single composite image with hardcoded 20-individual limits. When running larger populations ($N=25+$), matrix rows clipped, labels misaligned, parent lineages were difficult to trace across cluttered badges on lines, and sharing results required handling multiple disconnected PNGs.
- **Alternatives Considered:**
  1. *Keep single large image for Plot 1:* Remains squashed and unreadable as population and generation counts grow.
  2. *Interactive web app with Dash/Streamlit:* Requires hosting local servers and extra runtime dependencies.
  3. *Standalone High-Res Plots + Self-Contained Base64 Embedded HTML Dashboard:*
     - Separate Plot 1 into standalone high-visibility figures: `plot1a_individual_lineage.png`, `plot1b_population_composition.png`, `plot1c_fitness_dispersion.png`.
     - Dynamically scale Plot 1A height and axis limits according to detected population size (`pop_size = max(N)`).
     - Directly connect each child to its true parent(s) (1 solid line for clones, 2 dashed lines for recombinations) without line badges.
     - Embed the mutation symbol (`⚡`) directly inside the candidate's marker icon.
     - Boost Plot 4 (17-gene grid) resolution to 300 DPI (`figsize=(22, 25)`).
     - Package all plots into a standalone, portable `evolution_dashboard.html` with Base64 embedded images, summary statistics, and one-click PNG download buttons.
- **Decision:** Implement the standalone plot architecture, dynamic population scaling, in-symbol mutation markers, 300 DPI Plot 4 resolution, and the self-contained HTML dashboard in [`plot_evolution.py`](file:///home/nycolas/m/plot_evolution.py).
- **Rationale:** Eliminates layout clipping for arbitrary population sizes, makes lineages and parentage instantly traceable, provides crystal clear gene convergence plots, and yields a single shareable HTML file that can be distributed via email/slack with zero broken links.

---

## Decision 20: Safe Checkpoint Resumption of 30-Generation x 25-Individual GA Run
- **Date:** 2026-09-22
- **Context:** An extended evolutionary run (`--pop_size 25 --generations 30 --timesteps 300000 --concurrency 2 --envs_per_worker 20 --base_model base.zip --output_dir ga_results --n_best 5 --mutation_rate 0.02 --n_blend_genes 3`) was interrupted during Generation 25 after completing 24 full generations and partial evaluation of Generation 25 (`ind_00` and `ind_04` completed).
- **Alternatives Considered:**
  1. *Purge Generation 25 entirely:* Wastes valid evaluations of elite candidate policies.
  2. *Resume directly utilizing existing checkpoint caching:* `evolution.py` loads `evolution_history.json` (Generations 0–24) and leverages per-individual checkpoint caching (`model.zip` and `metrics.json`) to skip already evaluated individuals (`ind_00` and `ind_04`) while evaluating remaining candidates.
- **Decision:** Verified integrity of existing `model.zip` checkpoints, removed empty candidate directories (`ind_01`, `ind_05`), and resumed execution via Python 3.12 with identical hyperparameter flags.
- **Rationale:** Guarantees deterministic continuity from Generation 24 history, saves redundant GPU compute on completed individuals, and seamlessly executes the remaining generations (25–29) through completion.

---

## Decision 21: Post-30-Generation Architectural Diagnosis: Hierarchical Mutation & Penalized Effective Steps Fitness
- **Date:** 2026-09-22
- **Context:** Analysis of the completed 30-generation run revealed that 8/17 genes converged tightly, but individuals appeared to fluctuate across generation ranks in Plot 1A, recombined offspring frequently carried mutation badges (due to per-gene mutation without an individual-level gate), and the champion favored reckless 66% SR speed over high-reliability policies (e.g. 83%–87% SR).
- **Decisions & Recommendations:**
  1. *Hierarchical Mutation:* Introduce an individual-level mutation probability ($P_{\text{ind}} \approx 0.20$) before mutating genes, eliminating spurious mutation on every recombined offspring.
  2. *Penalized Timesteps Fitness:* Transition the objective function from pure $\bar{T}_{\text{goal}}$ (with unweighted 55% SR gate) to Penalized Effective Timesteps ($\mathcal{F} = \frac{1}{N}\sum (T_i \text{ if success else } 500)$).
  3. *High-SR Candidates:* Identify and surface existing high-reliability champions from the dataset: [`Gen 23 Ind 04`](file:///home/nycolas/m/ga_results/gen_23/ind_04/model.zip) (83.0% SR, 142.3 steps) and [`Gen 06 Ind 00`](file:///home/nycolas/m/ga_results/gen_06/ind_00/model.zip) (87.0% SR, 147.3 steps).
- **Rationale:** Aligns evolutionary pressure directly with the Pareto frontier of both high speed and high mission completion reliability while dampening phenotypic variance.

---

## Decision 22: Implementation of Hierarchical Individual Mutation, Dynamic Annealed Schedules, and Penalized Fitness
- **Date:** 2026-09-22
- **Context:** Following empirical diagnosis of the 30-generation run, the user requested:
  1. Switching mutation chance from per-gene to per-individual (gated by individual selection).
  2. If selected, a 50% chance of mutating 1 gene and 50% chance of mutating 2 genes.
  3. Linear decay of mutation perturbation band from $\pm 100\%$ at Generation 0 down to $\pm 25\%$ in final generations.
  4. Dynamic individual mutation rate increasing linearly from $0.10 \to 0.20$ as generations advance.
  5. Dynamic viability threshold ramping linearly from $50\% \to 65\%$ across generations, preserving dead individual replacement via survivor recombination (Group 1).
  6. Penalized effective timesteps fitness treating collisions and timeouts as $T_{\max} = 500$ steps.
- **Alternatives Considered:**
  1. *Unbounded Gaussian noise with static parameters:* Causes excessive destabilization of near-optimal alleles in late generations.
  2. *Static 55% viability threshold:* Too strict for early warm-start exploration, yet too permissive in late generations, allowing low-reliability candidates to survive.
  3. *Hierarchical Mutation + Dynamic Annealing + Penalized Timesteps:*
     - $P_{\text{ind}}(g) = 0.10 + 0.10 \cdot \frac{g}{G-1}$: Low early mutation rate lets elite crossover dominate; rising to 20% in late generations injects exploration against stagnation.
     - $B(g) = 1.00 - 0.75 \cdot \frac{g}{G-1}$: High $\pm 100\%$ early jumps escape local minima; fine $\pm 25\%$ late adjustments protect converged genomes.
     - $V(g) = 0.50 + 0.15 \cdot \frac{g}{G-1}$: Gentle 50% gate early on, rising to 65% as policies mature.
     - $\mathcal{F}_{\text{effective}} = \frac{1}{N} \sum (\text{steps if success else } 500)$: Seamlessly aligns optimization toward high success rate AND rapid flight.
- **Decision:** Implemented into [`evolution.py`](file:///home/nycolas/m/evolution.py) and [`plot_evolution.py`](file:///home/nycolas/m/plot_evolution.py).
- **Rationale:** Mathematically guides evolution toward the Pareto frontier of safety and speed, prevents mutation clutter, and dynamically tunes exploration vs. exploitation over the generational lifecycle.

---

## Decision 23: Expansion of Gene Bounds for Behavioral Diversity and Holonomic Capability
- **Date:** 2026-09-22
- **Context:** Analysis of the completed 30-generation evolutionary run revealed boundary saturation on several critical genes (e.g. `progress_multiplier` pinned against its previous upper limit of 25.0, `direction_factor_min` constrained at 0.50, `collision_penalty` capped at 10.0, and `retreat_penalty` strictly positive). To broaden phenotypic exploration and permit true omnidirectional locomotion without artificial heading biases, the search space bounds were widened.
- **Key Modifications to [`GENE_BOUNDS`](file:///home/nycolas/m/evolution.py#L22-L40):**
  1. `progress_multiplier`: $(2.0, 25.0) \to (1.0, 35.0)$ — allows substantially stronger forward pull to overcome loitering in open corridors.
  2. `direction_factor_min`: $(0.0, 0.5) \to (0.01, 1.0)$ — upper bound of 1.0 eliminates forward-facing requirements, granting equal reward for pure holonomic sideways and backward translation.
  3. `retreat_penalty`: $(0.005, 0.10) \to (-0.01, 0.01)$ — permits slightly negative penalties, allowing exploratory backward steps out of dead ends or traps without instant punitive losses.
  4. `collision_penalty`: $(0.5, 10.0) \to (0.1, 15.0)$ & `crash_speed_penalty`: $(0.5, 10.0) \to (0.1, 15.0)$ — provides greater dynamic range to penalize high-speed collisions severely while exploring softer boundaries.
  5. `wall_proximity_penalty`: $(0.01, 0.20) \to (0.001, 0.40)$ — accommodates both corridor-hugging aggressive shortcuts and wide-margin conservative policies.
  6. `goal_reward`: $(50.0, 200.0) \to (25.0, 250.0)$ & `early_goal_multiplier`: $(0.02, 0.50) \to (0.01, 0.75)$ — broadens the Pareto incentive between arrival safety and sprint urgency.
- **Alternatives Considered:**
  1. *Retaining narrow bounds:* Risks locking the population in suboptimal local attractor zones where genes remain pinned to boundary clamps.
  2. *Unbounded real-valued optimization:* Leads to reward explosion and numerical gradient instability in SAC actor-critic updates.
- **Decision:** Expanded the 17-gene bounds in both [`evolution.py`](file:///home/nycolas/m/evolution.py#L22-L40) and [`plot_evolution.py`](file:///home/nycolas/m/plot_evolution.py#L18-L36).
- **Rationale:** Heightens population genetic diversity, eliminates artificial saturation against boundary walls, and empowers the genetic algorithm to discover novel behavioral niches (e.g. holonomic sidestepping and aggressive corridor navigation).

---

## Decision 24: Vectorized 1,000-Episode Policy Evaluation & Exact Clone Checkpoint Inheritance
- **Date:** 2026-09-23
- **Context:** Observation of the `ga_results_v2` run revealed that while reward genes converged tightly in Plot 4, individual ranking trajectories in Plot 1A still exhibited zig-zagging jitter across generations. This stemmed from two sources: (1) evaluation measurement noise under 100 episodes ($\text{SEM} \approx \pm 6.0\text{ steps}$ when candidates differed by only $2\text{--}4\text{ steps}$), and (2) unnecessary retraining of unmutated elite clones from `base.zip`, introducing $\pm 10\text{--}15$ step stochastic RL training variance on identical genomes.
- **Alternatives Considered:**
  1. *Double RL training steps ($400\text{k} \to 800\text{k}$):* High compute penalty (+100% training time, ~60 hours per run) with negligible noise reduction since SAC policy entropy and mini-batch stochasticity remain.
  2. *Sequential 1,000-episode evaluation:* Takes ~5.3 minutes per individual on a single environment, ballooning total GA run time.
  3. *Vectorized 1,000-Episode Evaluation + Exact Clone Checkpoint Inheritance:*
     - Vectorize [`evaluate_policy`](file:///home/nycolas/m/evolution.py#L409) across 20 parallel worker environments via `DummyVecEnv`, completing 1,000 evaluation episodes in only $\sim 14\text{ seconds}$ ($\approx 71.9\text{ eps/s}$).
     - Detect exact clones (`len(parent_ids) == 1 and not is_mutated`) in [`train_individual_worker`](file:///home/nycolas/m/evolution.py#L482). Exact clones directly inherit the parent's `model.zip`, metrics, and fitness without retraining.
     - Offspring created via mutation (`is_mutated == True`) or recombination (`len(parent_ids) == 2`) must undergo full 400k-step retraining.
- **Decision:** Implement vectorized 1,000-episode evaluation and exact clone checkpoint inheritance in [`evolution.py`](file:///home/nycolas/m/evolution.py).
- **Rationale:**
  1. Reduces evaluation Standard Error of the Mean ($\text{SEM} = \sigma / \sqrt{N}$) from $\pm 6.0\text{ steps}$ to $\pm 1.89\text{ steps}$, ensuring a clear, statistically sound winner.
  2. Guarantees true phenotypic elitism: incumbents retain their rank and cannot regress due to training noise; they are only dethroned when an offspring genuinely beats their performance.
  3. Eliminates retraining for $\sim 25\%\text{--}35\%$ of the population each generation, accelerating GA generational throughput.

---

## Decision 25: Prevention of Multiprocessing `resource_tracker` Leaked Semaphores
- **Date:** 2026-09-23
- **Context:** At shutdown of extended evolutionary runs, Python raised: `UserWarning: resource_tracker: There appear to be 60 leaked semaphore objects to clean up at shutdown`. Investigation revealed that in Python 3.12, using `with ctx.Pool(processes=self.concurrency) as pool:` invokes `Pool.__exit__`, which executes `self.terminate()` without calling `pool.join()`. This abruptly aborts worker processes before their IPC handles can send unregister commands to the `resource_tracker` process. Over 30 generations with concurrency 2, exactly $30 \times 2 = 60$ worker processes were prematurely killed.
- **Alternatives Considered:**
  1. *Suppress warning via filterwarnings:* Masks the symptom without resolving the leaked OS handles.
  2. *Single-threaded execution:* Halves evolutionary throughput.
  3. *Proper Pool Lifecycle Management & Native PyTorch Multiprocessing:*
     - Switch imports to `torch.multiprocessing as mp`.
     - Configure `mp.set_sharing_strategy("file_system")` at module level to use temporary file descriptors rather than POSIX semaphores.
     - Replace `with ctx.Pool(...) as pool:` with an explicit `try ... finally: pool.close(); pool.join()`.
     - Add explicit resource cleanup before worker return (`del model`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`, `gc.collect()`).
- **Decision:** Implement explicit `try ... finally: pool.close(); pool.join()` pool lifecycle management and `file_system` sharing strategy in [`evolution.py`](file:///home/nycolas/m/evolution.py).
- **Rationale:**
  1. `pool.close()` gracefully prevents new tasks from being accepted while allowing worker queues to drain.
  2. `pool.join()` waits for all worker processes and handler threads to terminate cleanly, guaranteeing that all IPC semaphores are unregistered from the `resource_tracker`.
  3. Completely eliminates leaked semaphores and shutdown warnings (verified with 0 leaked objects).

---

## Decision 26: Elimination of Non-Elite Clone Skipping via Tournament Recombination & Universal Mutation
- **Date:** 2026-09-24
- **Context:** In `ga_results_f2`, exactly 19 out of 25 individuals were observed skipping retraining in every generation, severely suppressing genetic diversity. Investigation revealed that Group 4 ("tournament survivors") was originally configured to clone single non-elite survivors verbatim with hardcoded `is_mutated=False`. Consequently:
  1. Group 4 was completely exempt from mutation chance, rendering increases to `mutation_rate` ineffective for 60% of the population.
  2. Because Group 4 candidates had a single parent and were unmutated (`len(parent_ids) == 1 and not is_mutated`), the clone detector classified all 15 Group 4 individuals as exact clones alongside the 4 elites ($4 + 15 = 19$), skipping their retraining and freezing 76% of the population across generations.
- **Alternatives Considered:**
  1. *Apply mutation to single-parent Group 4 clones:* Still leaves ~80% of Group 4 unmutated (at $P_{\text{mut}} = 0.20$), resulting in 16/25 individuals skipping retraining and carrying forward redundant non-elite genomes.
  2. *Retrain unmutated single-parent non-elites:* Spends millions of compute steps retraining identical genomes that already lost to the elites, yielding no genetic diversity.
  3. *Tournament Crossover + Universal Mutation Chance + Strict Elite Retraining Bypass:*
     - Convert Group 4 into **Tournament Recombination**: each non-elite slot is produced by crossing over Parent 1 and Parent 2 (selected via Tournament $k=3$) with 3-gene blend crossover.
     - Subject **all individuals besides the top elites** (Groups 1, 3, and 4) to the individual mutation chance.
     - Restrict the retraining bypass strictly to true unmutated elites: `is_exact_clone = (ind.generation > 0 and ind.origin == "elite" and not ind.is_mutated)`.
- **Decision:** Implement Tournament Recombination for Group 4, subject all non-elite offspring to mutation, and strictly restrict retraining bypass to true elites in [`evolution.py`](file:///home/nycolas/m/evolution.py).
- **Rationale:**
  1. Preserves true phenotypic elitism: only the top $n_{\text{best}}$ incumbents skip retraining to guard against regression.
  2. Guarantees that 100% of non-elite slots ($20\text{--}21$ out of $25$) are brand-new recombined genomes that actively explore the search space.
  3. Ensures that every non-elite individual is subjected to the scheduled mutation chance, fully restoring genetic diversity.

---

## Decision 27: Strict Tier Separation in Genetic Reproduction (Non-Elite × Non-Elite Isolation)
- **Date:** 2026-09-27
- **Context:** In Group 4 tournament selection, Parent 2 was originally sampled from `survivors` (all survivors, including elites). When Parent 2 was an elite, unguided elite $\times$ non-elite recombination occurred, diluting elite lineages and contravening the multi-tier architectural specification.
- **Alternatives Considered:**
  1. *Allow unrestricted crossover across tiers:* Leads to genetic drift where top elite traits are diluted into the general non-elite population without directional selection.
  2. *Strict Tier Isolation:* Group 3 is strictly Elite $\times$ Elite; Group 4 is strictly Non-Elite $\times$ Non-Elite; Group 1 dead replacements draw strictly from non-elites (or immigrants).
- **Decision:** Enforce strict tier isolation in [`evolution.py`](file:///home/nycolas/m/evolution.py): draw both Parent 1 and Parent 2 for Group 4 and Group 1 exclusively from `candidate_pool` (`survivors[actual_n_best:]`).
- **Rationale:** Preserves elite purity in Group 3 while ensuring that non-elites recombine independently to foster bottom-up diversity.

---

## Decision 28: Scalable Visual Analytics with Dynamic Viability Schedules & Milestone Annotations
- **Date:** 2026-09-27
- **Context:** With 30 generations and 25 individuals/gen (750 total individuals), [`plot_evolution.py`](file:///home/nycolas/m/plot_evolution.py) suffered from two major flaws:
  1. It assumed a static $55\%$ survival threshold, failing to reflect the dynamic viability gate schedule ($50\% \to 65\%$).
  2. It annotated every single node and data point with text bounding boxes ($750$ individual text labels in Plot 1A, $30$ boxes per subplot in Plots 2 and 3), producing unreadable, colliding black text smears.
- **Alternatives Considered:**
  1. *Widen the figure resolution infinitely:* Increases file sizes to hundreds of megabytes without solving cognitive overload.
  2. *Remove all annotations entirely:* Makes it hard to identify key milestones without opening CSV/JSON files.
  3. *Dynamic Threshold Curves + Milestone-Only Annotations:* Plot the actual generational viability threshold schedule (`thresh_schedule`), suppress per-node text in Plot 1A (labeling only the generational champion), and restrict annotations in Plots 2, 3, and 4 strictly to milestones (G00, every 5 gens, final gen, and all-time records).
- **Decision:** Implement dynamic viability curves and milestone-only annotations in [`plot_evolution.py`](file:///home/nycolas/m/plot_evolution.py).
- **Rationale:** Produces publication-grade, uncluttered visual dashboards that scale gracefully to arbitrary generation counts without text collisions.














