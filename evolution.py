import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
import torch.multiprocessing as mp
import numpy as np
import copy
import itertools

if hasattr(mp, "set_sharing_strategy"):
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_training import OmniDroneEnv, DEFAULT_REWARD_PARAMS
from train import set_learning_rate

# ==============================================================================
# Genetic Algorithm Search Space & Bounds (17 Genes)
# Note: retreat_tolerance (0.05m) and retreat_window_steps (10 steps) are fixed values.
# ==============================================================================
GENE_BOUNDS = {
    "living_cost": (0.001, 0.075, float),
    "collision_penalty": (0.1, 15.0, float),
    "crash_speed_penalty": (0.1, 15.0, float),
    "high_speed_crash_penalty": (1.0, 15.0, float),
    "wall_proximity_penalty": (0.001, 0.40, float),
    "progress_multiplier": (1.0, 35.0, float),
    "stagnation_penalty": (0.005, 0.1, float),
    "loitering_penalty": (0.01, 0.75, float),
    "high_speed_reward": (0.001, 0.20, float),
    "overspeed_penalty": (0.01, 0.30, float),
    "constant_vel_reward": (0.001, 0.15, float),
    "goal_reward": (25.0, 250.0, float),
    "early_goal_multiplier": (0.01, 0.75, float),
    "fov_bonus": (0.001, 0.1, float),
    "direction_factor_min": (0.01, 1.0, float),
    "retreat_penalty": (-0.01, 0.01, float),
    "retreat_multiplier": (0.5, 10.0, float)
}

def generate_random_genes() -> dict:
    """Generates a random chromosome respecting defined parameter bounds."""
    genes = {}
    for key, (low, high, dtype) in GENE_BOUNDS.items():
        if dtype == int:
            genes[key] = int(np.random.randint(low, high + 1))
        else:
            genes[key] = float(np.random.uniform(low, high))
    return genes

# ==============================================================================
# Individual Representation
# ==============================================================================
class Individual:
    """Represents a single candidate reward policy in the population."""
    def __init__(
        self,
        ind_id: int,
        generation: int,
        genes: dict = None,
        origin: str = "random_init",
        parent_ids: list[int] = None,
        is_mutated: bool = False,
        mutated_genes: list[str] = None
    ):
        self.id = ind_id
        self.generation = generation
        self.genes = genes if genes is not None else generate_random_genes()
        self.origin = origin         # "random_init", "elite", "elite_recomb", "random_recomb", "tournament_selected", "random_fresh"
        self.parent_ids = parent_ids if parent_ids is not None else []
        self.is_mutated = is_mutated
        self.mutated_genes = mutated_genes if mutated_genes is not None else []
        self.fitness = None          # Average timesteps to goal across successful episodes
        self.is_alive = False        # Alive only if success_rate >= viability_threshold (55%)
        self.metrics = {}
        self.model_path = None
        self.training_time = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "generation": self.generation,
            "origin": self.origin,
            "parent_ids": self.parent_ids,
            "is_mutated": self.is_mutated,
            "mutated_genes": self.mutated_genes,
            "genes": self.genes,
            "fitness": self.fitness,
            "is_alive": self.is_alive,
            "metrics": self.metrics,
            "model_path": self.model_path,
            "training_time": self.training_time
        }

    @classmethod
    def from_dict(cls, data: dict):
        ind = cls(
            ind_id=data["id"],
            generation=data["generation"],
            genes=data["genes"],
            origin=data.get("origin", "random_init"),
            parent_ids=data.get("parent_ids", []),
            is_mutated=data.get("is_mutated", False),
            mutated_genes=data.get("mutated_genes", [])
        )
        ind.fitness = data.get("fitness")
        ind.is_alive = data.get("is_alive", False)
        ind.metrics = data.get("metrics", {})
        ind.model_path = data.get("model_path")
        ind.training_time = data.get("training_time", 0.0)
        return ind

# ==============================================================================
# Evolutionary Operators (Multi-Group Reproduction Cycle)
# ==============================================================================

def recombine_parents(parent1_genes: dict, parent2_genes: dict, n_blend_genes: int = 3) -> tuple[dict, dict]:
    """
    Recombines two parent genomes to produce 2 offspring.
    Blends exactly `n_blend_genes` (default 3) randomly chosen genes via arithmetic crossover:
      c1 = alpha * p1 + (1 - alpha) * p2
      c2 = (1 - alpha) * p1 + alpha * p2  (where alpha ~ Uniform(0, 1))
    The remaining genes undergo uniform discrete crossover (50% chance to swap).
    Clamps all values strictly within GENE_BOUNDS.
    """
    gene_keys = list(GENE_BOUNDS.keys())
    blend_keys = set(np.random.choice(gene_keys, size=min(n_blend_genes, len(gene_keys)), replace=False))

    child1_genes = {}
    child2_genes = {}

    for g in gene_keys:
        low, high, dtype = GENE_BOUNDS[g]
        p1_val = parent1_genes[g]
        p2_val = parent2_genes[g]

        if g in blend_keys:
            alpha = float(np.random.uniform(0.0, 1.0))
            c1_val = alpha * p1_val + (1.0 - alpha) * p2_val
            c2_val = (1.0 - alpha) * p1_val + alpha * p2_val
        else:
            if np.random.rand() < 0.5:
                c1_val, c2_val = p1_val, p2_val
            else:
                c1_val, c2_val = p2_val, p1_val

        if dtype == int:
            c1_val = int(np.clip(round(c1_val), low, high))
            c2_val = int(np.clip(round(c2_val), low, high))
        else:
            c1_val = float(np.clip(c1_val, low, high))
            c2_val = float(np.clip(c2_val, low, high))

        child1_genes[g] = c1_val
        child2_genes[g] = c2_val

    return child1_genes, child2_genes


def mutate_individual(genes: dict, mutation_band: float = 1.0) -> tuple[dict, list[str]]:
    """
    Applies targeted mutation to an individual selected for mutation.
    Has 50% chance of mutating exactly 1 gene, and 50% chance of mutating exactly 2 genes.
    Each mutated gene is shifted by +/- mutation_band (linearly decaying from 100% down to 25%).
    Direction: 50% chance increase by mutation_band * val, 50% chance decrease by mutation_band * val.
    Clamps strictly within GENE_BOUNDS.
    Returns the mutated genome and the list of mutated gene names.
    """
    mutated = copy.deepcopy(genes)
    gene_keys = list(GENE_BOUNDS.keys())

    # 50% chance of 1 gene, 50% chance of 2 genes
    num_genes_to_mutate = 1 if np.random.rand() < 0.5 else 2
    chosen_genes = list(np.random.choice(gene_keys, size=min(num_genes_to_mutate, len(gene_keys)), replace=False))
    mutated_genes = []

    for g in chosen_genes:
        g = str(g)
        low, high, dtype = GENE_BOUNDS[g]
        val = mutated[g]
        direction = 1 if np.random.rand() < 0.5 else -1
        if abs(val) > 1e-6:
            delta = mutation_band * abs(val) * direction
        else:
            delta = mutation_band * 0.1 * (high - low) * direction
        new_val = val + delta
        if dtype == int:
            mutated[g] = int(np.clip(round(new_val), low, high))
        else:
            mutated[g] = float(np.clip(new_val, low, high))
        mutated_genes.append(g)

    return mutated, mutated_genes


def mutate_genes(genes: dict, mutation_rate: float = 0.1, mutation_band: float = 0.5) -> tuple[dict, list[str]]:
    """Legacy compatibility helper."""
    return mutate_individual(genes, mutation_band=mutation_band)



def select_elites(survivors: list[Individual], num_elites: int = 4) -> list[Individual]:
    """Returns top `num_elites` surviving individuals (sorted by fitness)."""
    return survivors[:min(num_elites, len(survivors))]


def generate_next_generation(
    survivors: list[Individual],
    dead: list[Individual],
    next_gen: int,
    pop_size: int = 20,
    n_best: int = 4,
    current_mutation_rate: float = 0.1,
    current_mutation_band: float = 1.0,
    n_blend_genes: int = 3,
    tournament_size: int = 3,
    mutation_rate: float = None  # Legacy alias
) -> list[Individual]:
    """
    Executes the multi-group evolutionary reproduction cycle:
    - Group 1 (n_dead): Replaced by offspring produced by random recombination of survivors
      (subject to per-individual mutation selection: 50% 1 gene, 50% 2 genes with current_mutation_band).
    - Group 2 (n_best = 4): Top surviving elites pass directly (unmutated exact clones, skipping retraining).
    - Group 3 (C(n_best, 2) = 6): Half of the children from all-pairs elite recombination
      (subject to per-individual mutation selection).
    - Group 4 (n_left = pop_size - n_dead - n_best - C(n_best, 2)): Offspring produced by Tournament Selection Recombination
      (subject to per-individual mutation selection).

    Edge-case handling:
    If n_dead > pop_size - n_best - C(n_best, 2) (i.e. n_dead > 10, or n_survivors < 10):
    Prioritize survivors: take available elites (up to 4), elite offspring (up to 6), and fill
    remaining slots up to pop_size with random recombination offspring.
    If < 2 survive, fill deficits with fresh random individuals.
    """
    if mutation_rate is not None:
        current_mutation_rate = mutation_rate

    next_pop: list[Individual] = []

    def add_ind(genes: dict, origin: str, parent_ids: list[int] = None, is_mutated: bool = False, mutated_genes: list[str] = None):
        ind = Individual(
            ind_id=len(next_pop),
            generation=next_gen,
            genes=genes,
            origin=origin,
            parent_ids=parent_ids or [],
            is_mutated=is_mutated,
            mutated_genes=mutated_genes or []
        )
        next_pop.append(ind)

    n_survivors = len(survivors)
    n_dead = len(dead)

    # Edge case 1: Zero survivors (all died) -> Re-seed full population randomly
    if n_survivors == 0:
        print(f"[Reproduction Gen {next_gen:02d}] 0 survivors! Re-seeding entire population with fresh random genes.")
        while len(next_pop) < pop_size:
            add_ind(generate_random_genes(), origin="random_fresh")
        return next_pop

    # Edge case 2: Exactly 1 survivor -> Elite passes, cannot recombine, fill remainder randomly
    if n_survivors == 1:
        print(f"[Reproduction Gen {next_gen:02d}] 1 survivor! Passing 1 elite and re-seeding remainder randomly.")
        add_ind(copy.deepcopy(survivors[0].genes), origin="elite", parent_ids=[survivors[0].id])
        while len(next_pop) < pop_size:
            add_ind(generate_random_genes(), origin="random_fresh")
        return next_pop

    # Normal reproduction when n_survivors >= 2:
    # -------------------------------------------------------------
    # Group 2: Elites passing directly (unmutated)
    # -------------------------------------------------------------
    actual_n_best = min(n_best, n_survivors)
    elites = select_elites(survivors, actual_n_best)
    for e in elites:
        add_ind(copy.deepcopy(e.genes), origin="elite", parent_ids=[e.id], is_mutated=False)

    # -------------------------------------------------------------
    # Group 3: Elite Recombination Offspring
    # -------------------------------------------------------------
    # Form all pairs among elites: C(actual_n_best, 2)
    elite_pairs = list(itertools.combinations(elites, 2))
    elite_offspring_pool = []
    for p1, p2 in elite_pairs:
        c1, c2 = recombine_parents(p1.genes, p2.genes, n_blend_genes=n_blend_genes)
        elite_offspring_pool.append((c1, [p1.id, p2.id]))
        elite_offspring_pool.append((c2, [p1.id, p2.id]))

    # Total children = 2 * C(actual_n_best, 2). Rule: take randomly half of these sons (C(actual_n_best, 2))
    num_elite_offspring_needed = min(len(elite_pairs), pop_size - len(next_pop))
    np.random.shuffle(elite_offspring_pool)
    selected_elite_offspring = elite_offspring_pool[:num_elite_offspring_needed]
    for c_genes, parents in selected_elite_offspring:
        # Individual-level mutation selection
        if np.random.rand() < current_mutation_rate:
            mutated_genes, mut_names = mutate_individual(c_genes, mutation_band=current_mutation_band)
            add_ind(
                mutated_genes,
                origin="elite_recomb",
                parent_ids=parents,
                is_mutated=True,
                mutated_genes=mut_names
            )
        else:
            add_ind(
                c_genes,
                origin="elite_recomb",
                parent_ids=parents,
                is_mutated=False,
                mutated_genes=[]
            )

    # -------------------------------------------------------------
    # Slots remaining for Group 1 (dead replacement) and Group 4 (tournament-selected survivors)
    # -------------------------------------------------------------
    remaining_slots = pop_size - len(next_pop)

    # If n_dead <= remaining_slots, Group 1 gets n_dead and Group 4 gets the rest.
    # If n_dead > remaining_slots, Group 1 takes all remaining slots (Group 4 gets 0).
    if n_dead <= remaining_slots:
        target_group1 = n_dead
        target_group4 = remaining_slots - n_dead
    else:
        target_group1 = remaining_slots
        target_group4 = 0

    # -------------------------------------------------------------
    # Group 4: Tournament-Selected Recombination Offspring (Subject to Mutation)
    # -------------------------------------------------------------
    if target_group4 > 0:
        candidate_pool = list(survivors[actual_n_best:]) if len(survivors) > actual_n_best else list(survivors)
        if candidate_pool:
            for _ in range(target_group4):
                # Pick Parent 1 via Tournament Selection from non-elite survivors
                k1 = min(tournament_size, len(candidate_pool))
                sample1 = list(np.random.choice(candidate_pool, size=k1, replace=False))
                p1 = min(sample1, key=lambda ind: ind.fitness)

                # Pick Parent 2 via Tournament Selection strictly from non-elite survivors
                k2 = min(tournament_size, len(candidate_pool))
                sample2 = list(np.random.choice(candidate_pool, size=k2, replace=False))
                p2 = min(sample2, key=lambda ind: ind.fitness)

                # If both parents happen to be identical and candidate pool has > 1 member, pick another
                if p1.id == p2.id and len(candidate_pool) > 1:
                    other_candidates = [s for s in candidate_pool if s.id != p1.id]
                    if other_candidates:
                        k_other = min(tournament_size, len(other_candidates))
                        sample_other = list(np.random.choice(other_candidates, size=k_other, replace=False))
                        p2 = min(sample_other, key=lambda ind: ind.fitness)

                c_genes, _ = recombine_parents(p1.genes, p2.genes, n_blend_genes=n_blend_genes)

                # All individuals besides the elite are subjected to mutation chance
                if np.random.rand() < current_mutation_rate:
                    mutated_genes, mut_names = mutate_individual(c_genes, mutation_band=current_mutation_band)
                    add_ind(
                        mutated_genes,
                        origin="tournament_selected",
                        parent_ids=[p1.id, p2.id],
                        is_mutated=True,
                        mutated_genes=mut_names
                    )
                else:
                    add_ind(
                        c_genes,
                        origin="tournament_selected",
                        parent_ids=[p1.id, p2.id],
                        is_mutated=False,
                        mutated_genes=[]
                    )

    # -------------------------------------------------------------
    # Group 1: Random Recombination of Survivors (replacing dead individuals)
    # -------------------------------------------------------------
    while len(next_pop) < pop_size:
        # Pick 2 random distinct parents strictly from non-elites if available to maintain elite isolation
        parent_pool = candidate_pool if (candidate_pool and len(candidate_pool) >= 2) else survivors
        p1_idx, p2_idx = np.random.choice(len(parent_pool), size=2, replace=False)
        p1, p2 = parent_pool[p1_idx], parent_pool[p2_idx]
        c1, c2 = recombine_parents(p1.genes, p2.genes, n_blend_genes=n_blend_genes)

        if np.random.rand() < current_mutation_rate:
            c1_mutated, mut_names_1 = mutate_individual(c1, mutation_band=current_mutation_band)
            add_ind(
                c1_mutated,
                origin="random_recomb",
                parent_ids=[p1.id, p2.id],
                is_mutated=True,
                mutated_genes=mut_names_1
            )
        else:
            add_ind(
                c1,
                origin="random_recomb",
                parent_ids=[p1.id, p2.id],
                is_mutated=False,
                mutated_genes=[]
            )

        if len(next_pop) < pop_size:
            if np.random.rand() < current_mutation_rate:
                c2_mutated, mut_names_2 = mutate_individual(c2, mutation_band=current_mutation_band)
                add_ind(
                    c2_mutated,
                    origin="random_recomb",
                    parent_ids=[p1.id, p2.id],
                    is_mutated=True,
                    mutated_genes=mut_names_2
                )
            else:
                add_ind(
                    c2,
                    origin="random_recomb",
                    parent_ids=[p1.id, p2.id],
                    is_mutated=False,
                    mutated_genes=[]
                )


    # Group counts logging
    counts = {}
    for ind in next_pop:
        counts[ind.origin] = counts.get(ind.origin, 0) + 1
    print(f"[Reproduction Gen {next_gen:02d}] Next generation of {len(next_pop)} created: {counts}")

    assert len(next_pop) == pop_size, f"Expected {pop_size} individuals, got {len(next_pop)}"
    return next_pop

# ==============================================================================
# Fitness Evaluation
# ==============================================================================
def evaluate_policy(
    env: OmniDroneEnv | None,
    model: SAC,
    num_episodes: int = 1000,
    viability_threshold: float = 0.55,
    reward_params: dict = None,
    num_envs: int = 20
) -> tuple[float | None, dict, bool]:
    """
    Evaluates policy performance over num_episodes test tries (default: 1000).
    Vectorized across parallel environments for high-throughput evaluation (~14s for 1000 episodes).
    Fitness is defined as the average effective timesteps to goal,
    where collisions and timeouts are penalized as max_steps (500).
    An individual is only ranked and kept alive if success_rate >= viability_threshold.
    If success_rate < viability_threshold, the individual dies.
    """
    if reward_params is None and env is not None:
        reward_params = getattr(env, "reward_params", None)

    # Use vectorized environment for high-throughput multi-episode evaluation
    actual_num_envs = min(num_envs, num_episodes)
    def make_eval_env():
        return OmniDroneEnv(reward_params=reward_params)

    eval_vec_env = DummyVecEnv([make_eval_env for _ in range(actual_num_envs)])
    max_steps = 500

    successes, collisions, timeouts = 0, 0, 0
    goal_steps_list = []
    effective_steps_list = []
    speeds_list = []
    returns_list = []

    # Assign initial deterministic seeds to each environment slot
    env_seeds = [20000 + i for i in range(actual_num_envs)]
    next_ep_idx = actual_num_envs

    obs = eval_vec_env.reset()
    for i in range(actual_num_envs):
        obs[i] = eval_vec_env.envs[i].reset(seed=env_seeds[i])[0]

    ep_steps = np.zeros(actual_num_envs, dtype=int)
    ep_rewards = np.zeros(actual_num_envs, dtype=float)
    ep_speed_sums = np.zeros(actual_num_envs, dtype=float)
    completed_episodes = 0

    while completed_episodes < num_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_vec_env.step(actions)

        ep_steps += 1
        ep_rewards += rewards
        for i in range(actual_num_envs):
            ep_speed_sums[i] += eval_vec_env.envs[i].sim.speed

            if dones[i]:
                completed_episodes += 1
                info = infos[i]
                steps = int(ep_steps[i])
                reward = float(ep_rewards[i])
                avg_spd = float(ep_speed_sums[i] / max(1, steps))

                returns_list.append(reward)
                speeds_list.append(avg_spd)

                if info.get("is_success"):
                    successes += 1
                    goal_steps_list.append(steps)
                    effective_steps_list.append(steps)
                elif info.get("collision"):
                    collisions += 1
                    effective_steps_list.append(max_steps)
                else:
                    timeouts += 1
                    effective_steps_list.append(max_steps)

                # Reset env with next deterministic seed if more episodes needed
                if next_ep_idx < num_episodes:
                    obs[i] = eval_vec_env.envs[i].reset(seed=20000 + next_ep_idx)[0]
                    next_ep_idx += 1

                ep_steps[i] = 0
                ep_rewards[i] = 0.0
                ep_speed_sums[i] = 0.0

                if completed_episodes >= num_episodes:
                    break

    eval_vec_env.close()

    success_rate = successes / float(num_episodes)
    is_alive = bool(success_rate >= viability_threshold)

    avg_steps_to_goal = float(np.mean(goal_steps_list)) if goal_steps_list else float("inf")
    effective_steps = float(np.mean(effective_steps_list)) if effective_steps_list else float("inf")

    metrics = {
        "success_rate": success_rate,
        "collision_rate": collisions / float(num_episodes),
        "timeout_rate": timeouts / float(num_episodes),
        "avg_steps_to_goal": avg_steps_to_goal if goal_steps_list else None,
        "effective_steps_to_goal": effective_steps,
        "avg_speed": float(np.mean(speeds_list)) if speeds_list else 0.0,
        "avg_return": float(np.mean(returns_list)),
        "is_alive": is_alive,
        "viability_threshold": viability_threshold,
        "eval_episodes": num_episodes
    }

    fitness = effective_steps if is_alive else None
    return fitness, metrics, is_alive

# ==============================================================================
# Parallel Training Worker
# ==============================================================================
def train_individual_worker(args: tuple) -> dict:
    """
    Worker function executed in parallel subprocesses.
    Fine-tunes an individual from base.zip for timesteps under candidate reward genes.
    Exact clones (unmutated single-parent offspring) skip retraining and inherit parent checkpoint.
    """
    ind_data, config = args
    ind = Individual.from_dict(ind_data)

    gen_dir = Path(config["output_dir"]) / f"gen_{ind.generation:02d}"
    ind_dir = gen_dir / f"ind_{ind.id:02d}"
    ind_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = ind_dir / "model.zip"
    metrics_save_path = ind_dir / "metrics.json"

    # Check if already completed (supports safe resumption)
    if model_save_path.exists() and metrics_save_path.exists():
        with open(metrics_save_path, "r") as f:
            saved = json.load(f)
        ind.fitness = saved.get("fitness")
        ind.is_alive = saved.get("is_alive", False)
        ind.metrics = saved.get("metrics", {})
        ind.model_path = str(model_save_path)
        ind.training_time = saved.get("training_time", 0.0)
        status_str = f"ALIVE (Fitness: {ind.fitness:.1f} steps)" if ind.is_alive else f"DEAD (SR < {config.get('viability_threshold', 0.55)*100:.0f}%)"
        print(f"[Worker] Gen {ind.generation:02d} Ind {ind.id:02d} already finished. Status: {status_str}")
        return ind.to_dict()

    # Exact Clone Checkpoint Inheritance (skips retraining ONLY for unmutated elites)
    is_exact_clone = (ind.generation > 0 and ind.origin == "elite" and not ind.is_mutated)
    if is_exact_clone:
        parent_id = ind.parent_ids[0]
        parent_dir = Path(config["output_dir"]) / f"gen_{ind.generation - 1:02d}" / f"ind_{parent_id:02d}"
        parent_model_path = parent_dir / "model.zip"
        parent_metrics_path = parent_dir / "metrics.json"

        if parent_model_path.exists():
            import shutil
            shutil.copy2(parent_model_path, model_save_path)

            re_eval_needed = True
            if parent_metrics_path.exists():
                with open(parent_metrics_path, "r") as f:
                    parent_data = json.load(f)
                parent_metrics = copy.deepcopy(parent_data.get("metrics", {}))
                target_eval_eps = config.get("eval_episodes", 1000)

                # Inherit evaluation directly if already evaluated on identical episode count
                if parent_metrics.get("eval_episodes") == target_eval_eps:
                    re_eval_needed = False
                    viab_thresh = config.get("viability_threshold", 0.55)
                    success_rate = parent_metrics.get("success_rate", 0.0)
                    is_alive = bool(success_rate >= viab_thresh)

                    parent_metrics["viability_threshold"] = viab_thresh
                    parent_metrics["is_alive"] = is_alive
                    fitness = parent_metrics.get("effective_steps_to_goal") if is_alive else None

                    ind.fitness = fitness
                    ind.is_alive = is_alive
                    ind.metrics = parent_metrics
                    ind.model_path = str(model_save_path)
                    ind.training_time = 0.0

                    result_data = ind.to_dict()
                    with open(metrics_save_path, "w") as f:
                        json.dump(result_data, f, indent=2)

                    status_str = f"ALIVE (Fitness: {fitness:.1f} steps)" if is_alive else f"DEAD (SR: {success_rate*100:.1f}% < {viab_thresh*100:.0f}%)"
                    fit_str = f"{fitness:.1f} steps" if fitness is not None else "DEAD"
                    print(f"[Worker] Gen {ind.generation:02d} Ind {ind.id:02d} is an EXACT CLONE of Gen {ind.generation-1:02d} Ind {parent_id:02d} ({ind.origin}). SKIPPED RETRAINING! Status: {status_str}")
                    return ind.to_dict()

            if re_eval_needed:
                # Evaluate inherited model checkpoint once on target episodes (skips retraining)
                device = config["device"]
                model = SAC.load(str(model_save_path), device=device)
                viab_thresh = config.get("viability_threshold", 0.55)
                target_eval_eps = config.get("eval_episodes", 1000)
                num_envs = config.get("envs_per_worker", 20)
                fitness, metrics, is_alive = evaluate_policy(
                    env=None,
                    model=model,
                    num_episodes=target_eval_eps,
                    viability_threshold=viab_thresh,
                    reward_params=ind.genes,
                    num_envs=num_envs
                )
                ind.fitness = fitness
                ind.is_alive = is_alive
                ind.metrics = metrics
                ind.model_path = str(model_save_path)
                ind.training_time = 0.0

                result_data = ind.to_dict()
                with open(metrics_save_path, "w") as f:
                    json.dump(result_data, f, indent=2)

                status_str = f"ALIVE (Fitness: {fitness:.1f} steps)" if is_alive else f"DEAD (SR: {metrics['success_rate']*100:.1f}% < {viab_thresh*100:.0f}%)"
                print(f"[Worker] Gen {ind.generation:02d} Ind {ind.id:02d} is an EXACT CLONE of Gen {ind.generation-1:02d} Ind {parent_id:02d} ({ind.origin}). SKIPPED RETRAINING! Evaluated on {target_eval_eps} eps: {status_str}")
                
                try:
                    del model
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                import gc
                gc.collect()
                return ind.to_dict()

    print(f"\n[Worker] Starting Training: Gen {ind.generation:02d} | Ind {ind.id:02d}")
    start_time = time.time()

    # Create vectorized environment with individual's custom reward parameters
    num_envs = config["envs_per_worker"]
    def make_env():
        return OmniDroneEnv(reward_params=ind.genes)

    vec_env = DummyVecEnv([make_env for _ in range(num_envs)])

    # Warm-start fine-tuning from base model with train_freq=(4, 'step') and gradient_steps=4
    device = config["device"]
    base_model_path = config["base_model_path"]
    custom_objs = {
        "train_freq": (4, "step"),
        "gradient_steps": 4,
        "verbose": 0
    }
    model = SAC.load(base_model_path, env=vec_env, device=device, custom_objects=custom_objs)

    # Accelerate policy inference with torch.compile
    import torch
    if device == "cuda" and torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            model.policy.actor = torch.compile(model.policy.actor)
        except Exception:
            pass

    # Set elevated learning rate for rapid adaptation under novel reward shaping
    if config["learning_rate"]:
        set_learning_rate(model, config["learning_rate"])

    # Fine-tune for specified timesteps (300,000 / 400,000 / 500,000 steps)
    timesteps = config["timesteps_per_ind"]
    model.learn(total_timesteps=timesteps, progress_bar=False)

    # Unwrap compiled module before saving so checkpoints remain standard SB3 models
    if hasattr(model.policy.actor, "_orig_mod"):
        model.policy.actor = model.policy.actor._orig_mod

    # Save fine-tuned checkpoint
    model.save(str(model_save_path))
    vec_env.close()

    training_time = time.time() - start_time
    ind.training_time = training_time
    ind.model_path = str(model_save_path)

    # Evaluate trained policy over target episodes (vectorized across envs)
    viab_thresh = config.get("viability_threshold", 0.55)
    eval_eps = config.get("eval_episodes", 1000)
    fitness, metrics, is_alive = evaluate_policy(
        env=None,
        model=model,
        num_episodes=eval_eps,
        viability_threshold=viab_thresh,
        reward_params=ind.genes,
        num_envs=num_envs
    )

    ind.fitness = fitness
    ind.is_alive = is_alive
    ind.metrics = metrics

    # Save metrics and parameters
    result_data = ind.to_dict()
    with open(metrics_save_path, "w") as f:
        json.dump(result_data, f, indent=2)

    status_str = f"ALIVE (Fitness: {fitness:.1f} steps)" if is_alive else f"DEAD (SR: {metrics['success_rate']*100:.1f}% < {viab_thresh*100:.0f}%)"
    print(f"[Worker] Finished: Gen {ind.generation:02d} Ind {ind.id:02d} | Status: {status_str} | "
          f"Collisions: {metrics['collision_rate']*100:.1f}% | Time: {training_time/60:.1f}min")

    # Clean up PyTorch CUDA IPC memory and semaphores before process exit
    try:
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    import gc
    gc.collect()

    return ind.to_dict()

# ==============================================================================
# Genetic Algorithm Evolutionary Engine
# ==============================================================================
class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int = 20,
        num_generations: int = 20,
        timesteps_per_ind: int = 500000,
        concurrency: int = 2,
        envs_per_worker: int = 8,
        base_model_path: str = "base.zip",
        output_dir: str = "ga_results",
        learning_rate: float = 1e-3,
        eval_episodes: int = 1000,
        device: str = "cuda",
        n_best: int = 4,
        mutation_rate_start: float = 0.10,
        mutation_rate_end: float = 0.20,
        mutation_band_start: float = 1.00,
        mutation_band_end: float = 0.25,
        viability_threshold_start: float = 0.50,
        viability_threshold_end: float = 0.65,
        n_blend_genes: int = 3,
        tournament_size: int = 3,
        mutation_rate: float = None,
        viability_threshold: float = None
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.timesteps_per_ind = timesteps_per_ind
        self.concurrency = concurrency
        self.envs_per_worker = envs_per_worker
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.eval_episodes = eval_episodes
        self.device = device
        self.n_best = n_best
        self.n_blend_genes = n_blend_genes
        self.tournament_size = tournament_size

        # Dynamic schedules:
        # Mutation probability per recombined individual: 0.10 -> 0.20 (linear increase)
        if mutation_rate is not None:
            self.mutation_rate_start = mutation_rate
            self.mutation_rate_end = mutation_rate
        else:
            self.mutation_rate_start = mutation_rate_start
            self.mutation_rate_end = mutation_rate_end

        # Mutation perturbation band: +/-100% -> +/-25% (linear decrease)
        self.mutation_band_start = mutation_band_start
        self.mutation_band_end = mutation_band_end

        # Viability gate: 50% -> 65% (linear increase)
        if viability_threshold is not None:
            self.viability_threshold_start = viability_threshold
            self.viability_threshold_end = viability_threshold
        else:
            self.viability_threshold_start = viability_threshold_start
            self.viability_threshold_end = viability_threshold_end

        self.history_file = self.output_dir / "evolution_history.json"
        self.history = self._load_history()

    def _load_history(self) -> dict:
        if self.history_file.exists():
            with open(self.history_file, "r") as f:
                return json.load(f)
        return {"generations": []}

    def _save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def initialize_population(self, gen: int = 0) -> list[Individual]:
        """Initializes random population of individuals for generation 0."""
        return [Individual(ind_id=i, generation=gen, origin="random_init") for i in range(self.population_size)]

    def evaluate_generation_parallel(self, population: list[Individual], viability_threshold: float = 0.55) -> list[Individual]:
        """
        Evaluates population by training `concurrency` individuals at a time
        in parallel worker processes on GPU/CPU.
        """
        config = {
            "output_dir": str(self.output_dir),
            "base_model_path": self.base_model_path,
            "timesteps_per_ind": self.timesteps_per_ind,
            "envs_per_worker": self.envs_per_worker,
            "learning_rate": self.learning_rate,
            "eval_episodes": self.eval_episodes,
            "device": self.device,
            "viability_threshold": viability_threshold
        }

        tasks = [(ind.to_dict(), config) for ind in population]

        # Use PyTorch 'spawn' context with file_system sharing to prevent semaphore leaks at shutdown
        if hasattr(mp, "set_sharing_strategy"):
            try:
                mp.set_sharing_strategy("file_system")
            except Exception:
                pass
        ctx = mp.get_context("spawn")
        print(f"\n--- Launching Generation {population[0].generation} ({len(population)} individuals, concurrency={self.concurrency}) ---")
        pool = ctx.Pool(processes=self.concurrency)
        try:
            results = pool.map(train_individual_worker, tasks)
        finally:
            pool.close()
            pool.join()

        updated_population = [Individual.from_dict(res) for res in results]
        return updated_population

    def run(self):
        """Runs the evolutionary search loop across all generations."""
        print("=" * 75)
        print("GENETIC ALGORITHM: REWARD OPTIMIZATION FOR 2D OMNIDRONE")
        print(f"Population Size:         {self.population_size} individuals/generation")
        print(f"Total Generations:       {self.num_generations}")
        print(f"Fine-tune Steps/Ind:     {self.timesteps_per_ind:,} (total 800k with pre-train)")
        print(f"Evaluation Tries/Ind:    {self.eval_episodes} episodes")
        print(f"Viability Gate Schedule: {self.viability_threshold_start*100:.0f}% -> {self.viability_threshold_end*100:.0f}% (linear increase)")
        print(f"Individual Mutation:     {self.mutation_rate_start*100:.0f}% -> {self.mutation_rate_end*100:.0f}% chance (50% 1-gene, 50% 2-genes)")
        print(f"Gene Mutation Band:      +/- {self.mutation_band_start*100:.0f}% -> +/- {self.mutation_band_end*100:.0f}% (linear decrease)")
        print(f"Fitness Metric:          Penalized Effective Steps (Collisions/Timeouts = 500)")
        print(f"Elites Passing (n_best): {self.n_best}")
        print(f"Tournament Size:         {self.tournament_size}")
        print(f"Blend Genes Crossover:   {self.n_blend_genes} genes blended")
        print(f"Concurrency:             {self.concurrency} individuals in parallel on GPU")
        print(f"Envs per Worker:         {self.envs_per_worker}")
        print(f"Base Checkpoint:         {self.base_model_path}")
        print(f"Output Directory:        {self.output_dir}")
        print("=" * 75)

        # Resume from latest generation if history exists
        start_gen = len(self.history["generations"])
        population = None
        survivors = []
        dead = []

        for gen in range(start_gen, self.num_generations):
            gen_start_time = time.time()
            fraction = float(gen) / max(1, self.num_generations - 1)
            curr_mut_rate = self.mutation_rate_start + fraction * (self.mutation_rate_end - self.mutation_rate_start)
            curr_mut_band = self.mutation_band_start - fraction * (self.mutation_band_start - self.mutation_band_end)
            curr_viability = self.viability_threshold_start + fraction * (self.viability_threshold_end - self.viability_threshold_start)

            print(f"\n" + "=" * 75)
            print(f"GENERATION {gen:02d} / {self.num_generations:02d}")
            print(f"  Viability Gate:        success_rate >= {curr_viability*100:.1f}%")
            print(f"  Individual Mut Rate:   {curr_mut_rate*100:.1f}% per recombined offspring (50% 1-gene, 50% 2-genes)")
            print(f"  Gene Mutation Band:    +/- {curr_mut_band*100:.1f}%")
            print(f"  Fitness Formulation:   Penalized Effective Steps (Collisions/Timeouts = 500)")
            print("=" * 75)

            if gen == 0 or (population is None and start_gen == 0):
                population = self.initialize_population(gen=gen)
            elif population is None and start_gen > 0:
                # Resuming from previous generation history
                prev_gen_data = self.history["generations"][-1]
                prev_survivors = [Individual.from_dict(d) for d in prev_gen_data.get("ranked_survivors", [])]
                prev_dead = [Individual.from_dict(d) for d in prev_gen_data.get("dead_individuals", [])]
                population = generate_next_generation(
                    survivors=prev_survivors,
                    dead=prev_dead,
                    next_gen=gen,
                    pop_size=self.population_size,
                    n_best=self.n_best,
                    current_mutation_rate=curr_mut_rate,
                    current_mutation_band=curr_mut_band,
                    n_blend_genes=self.n_blend_genes,
                    tournament_size=self.tournament_size
                )
            else:
                # Multi-group evolutionary reproduction from current generation's survivors and dead
                population = generate_next_generation(
                    survivors=survivors,
                    dead=dead,
                    next_gen=gen,
                    pop_size=self.population_size,
                    n_best=self.n_best,
                    current_mutation_rate=curr_mut_rate,
                    current_mutation_band=curr_mut_band,
                    n_blend_genes=self.n_blend_genes,
                    tournament_size=self.tournament_size
                )

            # Train and evaluate individuals concurrently (2 at a time)
            population = self.evaluate_generation_parallel(population, viability_threshold=curr_viability)

            # Filter surviving individuals (success_rate >= curr_viability)
            survivors = [ind for ind in population if ind.is_alive]
            dead = [ind for ind in population if not ind.is_alive]

            # Rank ONLY individuals with success_rate >= curr_viability (sorted by lowest effective timesteps)
            survivors.sort(key=lambda ind: ind.fitness)

            gen_duration = time.time() - gen_start_time
            print(f"\n>>> Generation {gen:02d} Complete in {gen_duration/60:.1f}min <<<")
            print(f"Total Evaluated:         {len(population)}")
            print(f"Survivors (SR >= {curr_viability*100:.1f}%): {len(survivors)} / {len(population)}")
            print(f"Dead (SR < {curr_viability*100:.1f}%):       {len(dead)} / {len(population)}")

            if survivors:
                best_ind = survivors[0]
                print(f"\n--- Ranked Individuals (Success Rate >= {curr_viability*100:.1f}%) ---")
                for rank, ind in enumerate(survivors, start=1):
                    raw_steps = ind.metrics.get("avg_steps_to_goal")
                    raw_steps_str = f"{raw_steps:.1f}s" if raw_steps else "N/A"
                    print(f"  Rank #{rank:02d}: Ind {ind.id:02d} ({ind.origin}) | Effective Steps: {ind.fitness:.1f} (Goal: {raw_steps_str}) | "
                          f"SR: {ind.metrics['success_rate']*100:.1f}% | Collisions: {ind.metrics['collision_rate']*100:.1f}%")
                best_summary = {
                    "id": best_ind.id,
                    "origin": best_ind.origin,
                    "avg_steps_to_goal": best_ind.fitness,
                    "raw_steps_to_goal": best_ind.metrics.get("avg_steps_to_goal"),
                    "metrics": best_ind.metrics,
                    "genes": best_ind.genes
                }
            else:
                print(f"  [Warning] No individual reached the {curr_viability*100:.1f}% success threshold in this generation.")
                best_summary = None

            # Record generation summary
            gen_summary = {
                "generation": gen,
                "duration_seconds": gen_duration,
                "viability_threshold": curr_viability * 100.0,
                "mutation_rate": curr_mut_rate,
                "mutation_band": curr_mut_band,
                "total_evaluated": len(population),
                "num_survivors": len(survivors),
                "num_dead": len(dead),
                "best_individual": best_summary,
                "ranked_survivors": [ind.to_dict() for ind in survivors],
                "dead_individuals": [ind.to_dict() for ind in dead]
            }
            self.history["generations"].append(gen_summary)
            self._save_history()


            # Two-generation progression comparison
            if len(self.history["generations"]) >= 2:
                prev_g = self.history["generations"][-2]
                curr_g = self.history["generations"][-1]

                prev_steps = [s["fitness"] for s in prev_g.get("ranked_survivors", []) if s.get("fitness")]
                curr_steps = [s["fitness"] for s in curr_g.get("ranked_survivors", []) if s.get("fitness")]
                prev_srs = [s["metrics"]["success_rate"] for s in prev_g.get("ranked_survivors", [])]
                curr_srs = [s["metrics"]["success_rate"] for s in curr_g.get("ranked_survivors", [])]
                prev_colls = [s["metrics"]["collision_rate"] for s in prev_g.get("ranked_survivors", [])]
                curr_colls = [s["metrics"]["collision_rate"] for s in curr_g.get("ranked_survivors", [])]

                prev_mean_step = float(np.mean(prev_steps)) if prev_steps else 0.0
                curr_mean_step = float(np.mean(curr_steps)) if curr_steps else 0.0
                d_mean_step = curr_mean_step - prev_mean_step

                prev_best_step = prev_g["best_individual"]["avg_steps_to_goal"] if prev_g.get("best_individual") else 0.0
                curr_best_step = curr_g["best_individual"]["avg_steps_to_goal"] if curr_g.get("best_individual") else 0.0
                d_best_step = curr_best_step - prev_best_step

                prev_mean_sr = float(np.mean(prev_srs)) * 100.0 if prev_srs else 0.0
                curr_mean_sr = float(np.mean(curr_srs)) * 100.0 if curr_srs else 0.0
                d_sr = curr_mean_sr - prev_mean_sr

                prev_mean_coll = float(np.mean(prev_colls)) * 100.0 if prev_colls else 0.0
                curr_mean_coll = float(np.mean(curr_colls)) * 100.0 if curr_colls else 0.0
                d_coll = curr_mean_coll - prev_mean_coll

                print("\n" + "=" * 75)
                print(f"GENERATION PROGRESSION (Gen {prev_g['generation']:02d} -> Gen {curr_g['generation']:02d}):")
                print(f"  Mean Steps to Goal:  {prev_mean_step:.1f}  -->  {curr_mean_step:.1f}  (Δ {d_mean_step:+.1f} steps {'★' if d_mean_step < 0 else ''})")
                print(f"  Best Steps to Goal:  {prev_best_step:.1f}  -->  {curr_best_step:.1f}  (Δ {d_best_step:+.1f} steps {'★' if d_best_step < 0 else ''})")
                print(f"  Mean Success Rate:   {prev_mean_sr:.1f}%  -->  {curr_mean_sr:.1f}%  (Δ {d_sr:+.1f}% {'★' if d_sr > 0 else ''})")
                print(f"  Mean Collision Rate: {prev_mean_coll:.1f}%  -->  {curr_mean_coll:.1f}%  (Δ {d_coll:+.1f}% {'★' if d_coll < 0 else ''})")
                print(f"  Survivors / Dead:    {prev_g['num_survivors']}/{prev_g['num_dead']}  -->  {curr_g['num_survivors']}/{curr_g['num_dead']}")
                print("=" * 75)

            # Automatically update evolutionary plots after each generation
            try:
                from plot_evolution import (
                    plot_individual_existence,
                    plot_best_individual_metrics,
                    plot_mean_individual_metrics,
                    plot_gene_evolution
                )
                plot_individual_existence(self.history, self.output_dir / "plot1_individual_existence.png")
                plot_best_individual_metrics(self.history, self.output_dir / "plot2_best_individual_metrics.png")
                plot_mean_individual_metrics(self.history, self.output_dir / "plot3_mean_individual_metrics.png")
                plot_gene_evolution(self.history, self.output_dir / "plot4_gene_evolution.png")
                print(f"[Plots] Refreshed 4 evolutionary plots in: {self.output_dir}")
            except Exception as e:
                print(f"[Plots Warning] Could not update plots: {e}")

        # Final reporting: Find both best of final generation and all-time champion
        print("\n" + "=" * 75)
        print("EVOLUTIONARY SEARCH COMPLETE!")
        print(f"All results saved to: {self.output_dir.resolve()}")
        print(f"Evolutionary plots saved to: {self.output_dir.resolve()}/plot*.png")
        print("=" * 75)

        if self.history["generations"]:
            last_gen_data = self.history["generations"][-1]
            last_best = last_gen_data.get("best_individual")
            
            # Find all-time champion across all generations
            all_time_champ = None
            champ_gen = None
            for g in self.history["generations"]:
                for s in g.get("ranked_survivors", []):
                    if all_time_champ is None or s["fitness"] < all_time_champ["fitness"]:
                        all_time_champ = s
                        champ_gen = g["generation"]

            if last_best:
                print("\n" + "-" * 75)
                print(f"1. BEST INDIVIDUAL OF FINAL GENERATION (Gen {last_gen_data['generation']:02d}):")
                print(f"   ID: {last_best['id']} ({last_best.get('origin', 'N/A')}) | Fitness: {last_best['avg_steps_to_goal']:.1f} steps")
                m = last_best.get("metrics", {})
                print(f"   SR: {m.get('success_rate', 0.0)*100:.1f}% | Collisions: {m.get('collision_rate', 0.0)*100:.1f}% | Speed: {m.get('avg_speed', 0.0):.2f} m/s")
                print(f"   Reward Genes:")
                print(json.dumps(last_best.get("genes", {}), indent=4))

            if all_time_champ:
                print("\n" + "-" * 75)
                print(f"2. ALL-TIME EVOLUTIONARY CHAMPION (Across All Generations):")
                print(f"   Generation: {champ_gen:02d} | ID: {all_time_champ['id']} ({all_time_champ.get('origin', 'N/A')})")
                print(f"   Fitness:    {all_time_champ['fitness']:.1f} steps to goal")
                m = all_time_champ.get("metrics", {})
                print(f"   SR:         {m.get('success_rate', 0.0)*100:.1f}% | Collisions: {m.get('collision_rate', 0.0)*100:.1f}% | Speed: {m.get('avg_speed', 0.0):.2f} m/s")
                print(f"   Model Checkpoint: {all_time_champ.get('model_path')}")
                print(f"   Reward Genes:")
                print(json.dumps(all_time_champ.get("genes", {}), indent=4))
                print("-" * 75)

# ==============================================================================
# CLI Entrypoint
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm Reward Optimization for OmniDroneEnv")
    parser.add_argument("--pop_size", type=int, default=20, help="Individuals per generation (default: 20)")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations (default: 20)")
    parser.add_argument("--timesteps", type=int, default=500000, help="Timesteps per individual fine-tuning (default: 500k)")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of individuals to train in parallel (default: 2)")
    parser.add_argument("--envs_per_worker", type=int, default=8, help="Parallel envs per worker (default: 8)")
    parser.add_argument("--base_model", type=str, default="base.zip", help="Path to base model checkpoint")
    parser.add_argument("--output_dir", type=str, default="ga_results", help="Directory to save GA checkpoints and history")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for warm-start fine-tuning (default: 1e-3)")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="Evaluation tries per individual (default: 1000)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--n_best", type=int, default=4, help="Top elite individuals passing directly (default: 4)")
    parser.add_argument("--mutation_rate_start", type=float, default=0.20, help="Starting individual mutation rate (default: 0.10)")
    parser.add_argument("--mutation_rate_end", type=float, default=0.40, help="Ending individual mutation rate (default: 0.20)")
    parser.add_argument("--mutation_band_start", type=float, default=1.00, help="Starting mutation perturbation band (default: 1.0 = +/-100%%)")
    parser.add_argument("--mutation_band_end", type=float, default=0.25, help="Ending mutation perturbation band (default: 0.25 = +/-25%%)")
    parser.add_argument("--viability_start", type=float, default=0.50, help="Starting viability threshold (default: 0.50)")
    parser.add_argument("--viability_end", type=float, default=0.65, help="Ending viability threshold (default: 0.65)")
    parser.add_argument("--n_blend_genes", type=int, default=3, help="Number of genes to blend during crossover (default: 3)")
    parser.add_argument("--tournament_size", type=int, default=3, help="Tournament size for survivor selection (default: 3)")
    parser.add_argument("--mutation_rate", type=float, default=None, help="Fixed mutation rate override (optional)")
    parser.add_argument("--viability_threshold", type=float, default=None, help="Fixed viability threshold override (optional)")

    args = parser.parse_args()

    ga = GeneticAlgorithm(
        population_size=args.pop_size,
        num_generations=args.generations,
        timesteps_per_ind=args.timesteps,
        concurrency=args.concurrency,
        envs_per_worker=args.envs_per_worker,
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        eval_episodes=args.eval_episodes,
        device=args.device,
        n_best=args.n_best,
        mutation_rate_start=args.mutation_rate_start,
        mutation_rate_end=args.mutation_rate_end,
        mutation_band_start=args.mutation_band_start,
        mutation_band_end=args.mutation_band_end,
        viability_threshold_start=args.viability_start,
        viability_threshold_end=args.viability_end,
        n_blend_genes=args.n_blend_genes,
        tournament_size=args.tournament_size,
        mutation_rate=args.mutation_rate,
        viability_threshold=args.viability_threshold
    )

    ga.run()

