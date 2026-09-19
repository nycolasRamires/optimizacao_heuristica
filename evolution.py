import os
import sys
import json
import time
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_training import OmniDroneEnv, DEFAULT_REWARD_PARAMS
from train import set_learning_rate

# ==============================================================================
# Genetic Algorithm Search Space & Bounds (19 Genes)
# ==============================================================================
GENE_BOUNDS = {
    "living_cost": (0.001, 0.05, float),
    "collision_penalty": (0.5, 10.0, float),
    "crash_speed_penalty": (0.5, 10.0, float),
    "high_speed_crash_penalty": (1.0, 15.0, float),
    "wall_proximity_penalty": (0.01, 0.20, float),
    "progress_multiplier": (2.0, 25.0, float),
    "stagnation_penalty": (0.005, 0.05, float),
    "loitering_penalty": (0.02, 0.5, float),
    "high_speed_reward": (0.005, 0.10, float),
    "overspeed_penalty": (0.01, 0.20, float),
    "constant_vel_reward": (0.005, 0.10, float),
    "goal_reward": (50.0, 200.0, float),
    "early_goal_multiplier": (0.02, 0.50, float),
    "fov_bonus": (0.002, 0.05, float),
    "direction_factor_min": (0.0, 0.5, float),
    "retreat_tolerance": (0.02, 0.15, float),
    "retreat_penalty": (0.005, 0.10, float),
    "retreat_multiplier": (0.5, 10.0, float),
    "retreat_window_steps": (5, 25, int)
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
    def __init__(self, ind_id: int, generation: int, genes: dict = None):
        self.id = ind_id
        self.generation = generation
        self.genes = genes if genes is not None else generate_random_genes()
        self.fitness = None          # Average timesteps to goal across successful episodes
        self.is_alive = False        # Alive only if success_rate >= 0.5 (50%)
        self.metrics = {}
        self.model_path = None
        self.training_time = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "generation": self.generation,
            "genes": self.genes,
            "fitness": self.fitness,
            "is_alive": self.is_alive,
            "metrics": self.metrics,
            "model_path": self.model_path,
            "training_time": self.training_time
        }

    @classmethod
    def from_dict(cls, data: dict):
        ind = cls(ind_id=data["id"], generation=data["generation"], genes=data["genes"])
        ind.fitness = data.get("fitness")
        ind.is_alive = data.get("is_alive", False)
        ind.metrics = data.get("metrics", {})
        ind.model_path = data.get("model_path")
        ind.training_time = data.get("training_time", 0.0)
        return ind

# ==============================================================================
# Evolutionary Operator Placeholders (User to specify exact logic)
# ==============================================================================
def select_elites(population: list[Individual], num_elites: int) -> list[Individual]:
    """
    Placeholder for elitism policy.
    To be customized by the user.
    """
    pass

def recombine(parent1: Individual, parent2: Individual, child_id: int, generation: int) -> Individual:
    """
    Placeholder for crossover / recombination policy.
    To be customized by the user.
    """
    pass

def mutate(individual: Individual, mutation_rate: float = 0.1) -> Individual:
    """
    Placeholder for mutation policy.
    To be customized by the user.
    """
    pass

# ==============================================================================
# Fitness Evaluation
# ==============================================================================
def evaluate_policy(env: OmniDroneEnv, model: SAC, num_episodes: int = 100) -> tuple[float | None, dict, bool]:
    """
    Evaluates policy performance over 100 test tries.
    Fitness is defined as the average timesteps taken to reach the goal.
    An individual is only ranked and kept alive if success_rate >= 0.5 (50%).
    If success_rate < 0.5, the individual dies.
    """
    successes, collisions, timeouts = 0, 0, 0
    goal_steps_list = []
    speeds_list = []
    returns_list = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=20000 + ep)
        done = False
        ep_steps = 0
        ep_reward = 0.0
        ep_speeds = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1
            ep_reward += reward
            ep_speeds.append(env.sim.speed)
            done = terminated or truncated

        returns_list.append(ep_reward)
        if info.get("is_success"):
            successes += 1
            goal_steps_list.append(ep_steps)
        elif info.get("collision"):
            collisions += 1
        elif truncated:
            timeouts += 1

        if ep_speeds:
            speeds_list.append(float(np.mean(ep_speeds)))

    success_rate = successes / float(num_episodes)
    is_alive = bool(success_rate >= 0.5)

    avg_steps_to_goal = float(np.mean(goal_steps_list)) if goal_steps_list else float("inf")

    metrics = {
        "success_rate": success_rate,
        "collision_rate": collisions / float(num_episodes),
        "timeout_rate": timeouts / float(num_episodes),
        "avg_steps_to_goal": avg_steps_to_goal if goal_steps_list else None,
        "avg_speed": float(np.mean(speeds_list)) if speeds_list else 0.0,
        "avg_return": float(np.mean(returns_list)),
        "is_alive": is_alive
    }

    # Fitness is the average timesteps to reach the goal (only for individuals with success_rate >= 0.5)
    fitness = avg_steps_to_goal if is_alive else None

    return fitness, metrics, is_alive

# ==============================================================================
# Parallel Training Worker
# ==============================================================================
def train_individual_worker(args: tuple) -> dict:
    """
    Worker function executed in parallel subprocesses.
    Fine-tunes an individual from base.zip for 500k timesteps under candidate reward genes.
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
        status_str = f"ALIVE (Fitness: {ind.fitness:.1f} steps)" if ind.is_alive else "DEAD (SR < 50%)"
        print(f"[Worker] Gen {ind.generation:02d} Ind {ind.id:02d} already finished. Status: {status_str}")
        return ind.to_dict()

    print(f"\n[Worker] Starting Training: Gen {ind.generation:02d} | Ind {ind.id:02d}")
    start_time = time.time()

    # Create vectorized environment with individual's custom reward parameters
    num_envs = config["envs_per_worker"]
    def make_env():
        return OmniDroneEnv(reward_params=ind.genes)

    vec_env = DummyVecEnv([make_env for _ in range(num_envs)])

    # Warm-start fine-tuning from base model
    device = config["device"]
    base_model_path = config["base_model_path"]
    model = SAC.load(base_model_path, env=vec_env, device=device)

    # Set elevated learning rate for rapid adaptation under novel reward shaping
    if config["learning_rate"]:
        set_learning_rate(model, config["learning_rate"])

    # Fine-tune for specified timesteps (500,000 steps)
    timesteps = config["timesteps_per_ind"]
    model.learn(total_timesteps=timesteps, progress_bar=False)

    # Save fine-tuned checkpoint
    model.save(str(model_save_path))
    vec_env.close()

    training_time = time.time() - start_time
    ind.training_time = training_time
    ind.model_path = str(model_save_path)

    # Evaluate trained policy over 100 tries
    eval_env = OmniDroneEnv(reward_params=ind.genes)
    fitness, metrics, is_alive = evaluate_policy(eval_env, model, num_episodes=config["eval_episodes"])
    eval_env.close()

    ind.fitness = fitness
    ind.is_alive = is_alive
    ind.metrics = metrics

    # Save metrics and parameters
    result_data = ind.to_dict()
    with open(metrics_save_path, "w") as f:
        json.dump(result_data, f, indent=2)

    status_str = f"ALIVE (Avg Steps: {fitness:.1f})" if is_alive else f"DIED (SR: {metrics['success_rate']*100:.1f}% < 50%)"
    print(f"[Worker] Finished: Gen {ind.generation:02d} Ind {ind.id:02d} | Status: {status_str} | "
          f"Collisions: {metrics['collision_rate']*100:.1f}% | Time: {training_time/60:.1f}min")

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
        eval_episodes: int = 100,
        device: str = "cuda"
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
        """Initializes random population of individuals for a generation."""
        return [Individual(ind_id=i, generation=gen) for i in range(self.population_size)]

    def evaluate_generation_parallel(self, population: list[Individual]) -> list[Individual]:
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
            "device": self.device
        }

        tasks = [(ind.to_dict(), config) for ind in population]

        # Use 'spawn' multiprocessing context for clean CUDA isolation across workers
        ctx = mp.get_context("spawn")
        print(f"\n--- Launching Generation {population[0].generation} ({len(population)} individuals, concurrency={self.concurrency}) ---")
        
        with ctx.Pool(processes=self.concurrency) as pool:
            results = pool.map(train_individual_worker, tasks)

        updated_population = [Individual.from_dict(res) for res in results]
        return updated_population

    def run(self):
        """Runs the evolutionary search loop across all generations."""
        print("=" * 75)
        print("GENETIC ALGORITHM: REWARD OPTIMIZATION FOR 2D OMNIDRONE")
        print(f"Population Size:       {self.population_size} individuals/generation")
        print(f"Total Generations:     {self.num_generations}")
        print(f"Fine-tune Steps/Ind:   {self.timesteps_per_ind:,} (total 800k with pre-train)")
        print(f"Evaluation Tries/Ind:  {self.eval_episodes} episodes")
        print(f"Viability Threshold:   success_rate >= 0.50 (dies if < 50%)")
        print(f"Fitness Metric:        average timesteps to goal (lower is better)")
        print(f"Concurrency:           {self.concurrency} individuals in parallel on GPU")
        print(f"Envs per Worker:       {self.envs_per_worker}")
        print(f"Base Checkpoint:       {self.base_model_path}")
        print(f"Output Directory:      {self.output_dir}")
        print("=" * 75)

        # Resume from latest generation if history exists
        start_gen = len(self.history["generations"])
        population = None

        for gen in range(start_gen, self.num_generations):
            gen_start_time = time.time()

            if gen == 0 or population is None:
                population = self.initialize_population(gen=gen)
            else:
                # Evolutionary reproduction step:
                # 1. Separate surviving ranked individuals from dead individuals
                survivors = [ind for ind in population if ind.is_alive]
                dead = [ind for ind in population if not ind.is_alive]

                # User operator placeholders (currently pass)
                elites = select_elites(survivors, num_elites=2)

                # For individuals that died (success_rate < 0.5), random individuals take their place
                next_pop = []
                next_id = 0

                # If operators are defined and return individuals, keep them; otherwise replenish
                if elites:
                    for e in elites:
                        e_next = copy.deepcopy(e)
                        e_next.id = next_id
                        e_next.generation = gen
                        e_next.fitness = None
                        e_next.is_alive = False
                        next_pop.append(e_next)
                        next_id += 1

                # Fill remaining slots with new random individuals (replacing dead individuals)
                while len(next_pop) < self.population_size:
                    next_pop.append(Individual(ind_id=next_id, generation=gen, genes=generate_random_genes()))
                    next_id += 1

                population = next_pop

            # Train and evaluate individuals concurrently (2 at a time)
            population = self.evaluate_generation_parallel(population)

            # Filter surviving individuals (success_rate >= 0.5)
            survivors = [ind for ind in population if ind.is_alive]
            dead = [ind for ind in population if not ind.is_alive]

            # Rank ONLY individuals with success_rate >= 0.5 (sorted by lowest avg timesteps to goal)
            survivors.sort(key=lambda ind: ind.fitness)

            gen_duration = time.time() - gen_start_time
            print(f"\n>>> Generation {gen:02d} Complete in {gen_duration/60:.1f}min <<<")
            print(f"Total Evaluated:  {len(population)}")
            print(f"Survivors (SR >= 50%): {len(survivors)} / {len(population)}")
            print(f"Dead (SR < 50%):       {len(dead)} / {len(population)} (replaced with random individuals next gen)")

            if survivors:
                best_ind = survivors[0]
                print(f"\n--- Ranked Individuals (Success Rate >= 50%) ---")
                for rank, ind in enumerate(survivors, start=1):
                    print(f"  Rank #{rank:02d}: Ind {ind.id:02d} | Avg Steps to Goal: {ind.fitness:.1f} | "
                          f"SR: {ind.metrics['success_rate']*100:.1f}% | Collisions: {ind.metrics['collision_rate']*100:.1f}%")
                best_summary = {
                    "id": best_ind.id,
                    "avg_steps_to_goal": best_ind.fitness,
                    "metrics": best_ind.metrics,
                    "genes": best_ind.genes
                }
            else:
                print("  [Warning] No individual reached the 50% success threshold in this generation.")
                best_summary = None

            # Record generation summary
            gen_summary = {
                "generation": gen,
                "duration_seconds": gen_duration,
                "total_evaluated": len(population),
                "num_survivors": len(survivors),
                "num_dead": len(dead),
                "best_individual": best_summary,
                "ranked_survivors": [ind.to_dict() for ind in survivors],
                "dead_individuals": [ind.to_dict() for ind in dead]
            }
            self.history["generations"].append(gen_summary)
            self._save_history()

        print("\n" + "=" * 75)
        print("EVOLUTIONARY SEARCH COMPLETE!")
        print(f"All results saved to: {self.output_dir.resolve()}")
        print("=" * 75)

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
    parser.add_argument("--eval_episodes", type=int, default=100, help="Evaluation tries per individual (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

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
        device=args.device
    )

    ga.run()
