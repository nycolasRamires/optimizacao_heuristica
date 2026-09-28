import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for plot saving
import matplotlib.pyplot as plt

import torch
from stable_baselines3 import SAC
from rl_training import OmniDroneEnv
from evolution import evaluate_policy

def load_ga_history(results_dir: Path) -> dict:
    history_file = results_dir / "evolution_history.json"
    if not history_file.exists():
        raise FileNotFoundError(f"History file not found at: {history_file}")
    with open(history_file, "r") as f:
        return json.load(f)

def print_generation_summary_table(history: dict):
    generations = history.get("generations", [])
    if not generations:
        print("No generations recorded in history file.")
        return

    print("\n" + "=" * 95)
    print("                      GENETIC ALGORITHM EVOLUTIONARY PROGRESSION REPORT")
    print("=" * 95)
    header = f"{'Gen':>4} | {'Time':>7} | {'Surv/Dead':>9} | {'Best Ind':>10} | {'Origin':>13} | {'Avg Steps':>9} | {'SR (%)':>7} | {'Coll (%)':>8} | {'Speed (m/s)':>11}"
    print(header)
    print("-" * 95)

    for g in generations:
        gen_idx = g["generation"]
        duration_min = g["duration_seconds"] / 60.0
        surv_str = f"{g['num_survivors']}/{g['num_dead']}"
        best = g.get("best_individual")

        if best:
            ind_id = f"Ind {best['id']:02d}"
            origin = best.get("origin", "unknown")
            fitness_str = f"{best['avg_steps_to_goal']:.1f}" if best.get("avg_steps_to_goal") else "N/A"
            m = best.get("metrics", {})
            sr_str = f"{m.get('success_rate', 0.0) * 100:.1f}%"
            coll_str = f"{m.get('collision_rate', 0.0) * 100:.1f}%"
            spd_str = f"{m.get('avg_speed', 0.0):.2f}"
        else:
            ind_id = "None"
            origin = "-"
            fitness_str = "N/A"
            sr_str = "0.0%"
            coll_str = "100.0%"
            spd_str = "0.00"

        print(f"{gen_idx:4d} | {duration_min:6.1f}m | {surv_str:>9} | {ind_id:>10} | {origin:>13} | {fitness_str:>9} | {sr_str:>7} | {coll_str:>8} | {spd_str:>11}")

    print("=" * 95)

def find_champion_individual(history: dict, results_dir: Path) -> tuple[dict, str]:
    best_candidate = None
    best_gen = None

    for g in history.get("generations", []):
        ranked = g.get("ranked_survivors", [])
        if ranked:
            top_ind = ranked[0]
            if best_candidate is None or top_ind["fitness"] < best_candidate["fitness"]:
                best_candidate = top_ind
                best_gen = g["generation"]

    if best_candidate is None:
        return None, None

    model_path = results_dir / f"gen_{best_gen:02d}" / f"ind_{best_candidate['id']:02d}" / "model.zip"
    return best_candidate, str(model_path)

def generate_evolution_plots(history: dict, results_dir: Path, output_file: str = "evolution_report.png"):
    generations = history.get("generations", [])
    if not generations:
        return

    gen_indices = [g["generation"] for g in generations]
    best_fitness = []
    mean_fitness = []
    success_rates = []
    collision_rates = []
    timeout_rates = []
    survivors_count = [g["num_survivors"] for g in generations]
    dead_count = [g["num_dead"] for g in generations]

    # Gene tracking for critical parameters
    tracked_genes = ["collision_penalty", "progress_multiplier", "goal_reward", "overspeed_penalty", "high_speed_reward", "constant_vel_reward"]
    gene_trajectories = {k: [] for k in tracked_genes}

    for g in generations:
        best = g.get("best_individual")
        ranked = g.get("ranked_survivors", [])

        if best and best.get("avg_steps_to_goal") is not None:
            best_fitness.append(best["avg_steps_to_goal"])
            m = best.get("metrics", {})
            success_rates.append(m.get("success_rate", 0.0) * 100)
            collision_rates.append(m.get("collision_rate", 0.0) * 100)
            timeout_rates.append(m.get("timeout_rate", 0.0) * 100)
            for k in tracked_genes:
                gene_trajectories[k].append(best.get("genes", {}).get(k, np.nan))
        else:
            best_fitness.append(np.nan)
            success_rates.append(0.0)
            collision_rates.append(100.0)
            timeout_rates.append(0.0)
            for k in tracked_genes:
                gene_trajectories[k].append(np.nan)

        if ranked:
            all_steps = [ind["fitness"] for ind in ranked if ind.get("fitness") is not None]
            mean_fitness.append(np.mean(all_steps) if all_steps else np.nan)
        else:
            mean_fitness.append(np.nan)

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Genetic Algorithm Reward Optimization — Evolutionary History", fontsize=16, fontweight="bold", y=0.98)

    # Subplot 1: Fitness (Steps to Goal)
    ax1 = axs[0, 0]
    ax1.plot(gen_indices, best_fitness, marker="o", color="#1f77b4", linewidth=2, label="Best Individual (Fastest)")
    ax1.plot(gen_indices, mean_fitness, marker="s", linestyle="--", color="#aec7e8", linewidth=1.5, label="Surviving Population Mean")
    ax1.set_title("Fitness Convergence (Average Timesteps to Goal)", fontweight="bold")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Steps to Reach Goal (Lower is Better)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Navigation Reliability Rates
    ax2 = axs[0, 1]
    ax2.plot(gen_indices, success_rates, marker="^", color="#2ca02c", linewidth=2, label="Success Rate (%)")
    ax2.plot(gen_indices, collision_rates, marker="x", color="#d62728", linewidth=2, label="Collision Rate (%)")
    ax2.plot(gen_indices, timeout_rates, marker="d", color="#ff7f0e", linewidth=1.5, linestyle=":", label="Timeout Rate (%)")
    ax2.axhline(50.0, color="gray", linestyle="--", alpha=0.7, label="Viability Threshold (50%)")
    ax2.set_title("Reliability of Best Individual Across Generations", fontweight="bold")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(-2, 102)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Population Viability (Survivors vs Dead)
    ax3 = axs[1, 0]
    bars_surv = ax3.bar(gen_indices, survivors_count, color="#2ca02c", alpha=0.8, label="Survivors (SR >= 50%)")
    bars_dead = ax3.bar(gen_indices, dead_count, bottom=survivors_count, color="#d62728", alpha=0.8, label="Dead (SR < 50%)")
    ax3.set_title("Population Viability per Generation (Total N = 20)", fontweight="bold")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Number of Individuals")
    ax3.set_ylim(0, max(22, max(survivors_count[0] + dead_count[0], 20) + 2))
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Gene Trajectories of Top Individual
    ax4 = axs[1, 1]
    colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for idx, k in enumerate(tracked_genes):
        vals = gene_trajectories[k]
        # Normalize to [0, 1] relative to first gen or max for clean comparative display
        clean_vals = [v for v in vals if not np.isnan(v)]
        if clean_vals and max(clean_vals) > min(clean_vals):
            norm_vals = [(v - min(clean_vals)) / (max(clean_vals) - min(clean_vals)) for v in vals]
            ax4.plot(gen_indices, norm_vals, marker=".", label=f"{k} (rel: {clean_vals[-1]:.2f})", color=colors[idx % len(colors)])
        else:
            ax4.plot(gen_indices, vals, marker=".", label=f"{k}", color=colors[idx % len(colors)])
    ax4.set_title("Normalized Key Gene Drift (Best Individual)", fontweight="bold")
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Relative Parameter Drift")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = results_dir / output_file
    plt.savefig(str(plot_path), dpi=200)
    plt.close()
    print(f"\n[Visual Report] Evolution graph saved to: {plot_path.resolve()}")

def benchmark_champion_vs_baseline(champion_model_path: str, champion_genes: dict, base_model_path: str = "base.zip", num_episodes: int = 100):
    print("\n" + "=" * 80)
    print("         HEAD-TO-HEAD TOURNAMENT: BASE MODEL vs. GA CHAMPION")
    print(f"               Evaluated over {num_episodes} seeded deterministic episodes")
    print("=" * 80)

    # 1. Evaluate Baseline
    print(f"[Tournament] Evaluating Baseline ({base_model_path})...")
    base_env = OmniDroneEnv()
    base_model = SAC.load(base_model_path, env=base_env, device="cuda" if torch.cuda.is_available() else "cpu")
    base_fitness, base_metrics, base_alive = evaluate_policy(base_env, base_model, num_episodes=num_episodes)
    base_env.close()

    # 2. Evaluate GA Champion
    print(f"[Tournament] Evaluating GA Champion ({champion_model_path})...")
    champ_env = OmniDroneEnv(reward_params=champion_genes)
    champ_model = SAC.load(champion_model_path, env=champ_env, device="cuda" if torch.cuda.is_available() else "cpu")
    champ_fitness, champ_metrics, champ_alive = evaluate_policy(champ_env, champ_model, num_episodes=num_episodes)
    champ_env.close()

    # Print Comparative Table
    print("\n" + "-" * 80)
    print(f"{'Performance Metric':<28} | {'Baseline (base.zip)':<20} | {'GA Champion':<20} | {'Delta':<12}")
    print("-" * 80)

    def format_diff(val_c, val_b, unit="%", invert=False):
        diff = val_c - val_b
        sign = "+" if diff > 0 else ""
        if invert:
            good = diff < 0
        else:
            good = diff > 0
        status = "★" if good else " "
        return f"{sign}{diff:.1f}{unit} {status}"

    # Success rate
    b_sr = base_metrics['success_rate'] * 100
    c_sr = champ_metrics['success_rate'] * 100
    print(f"{'Success Rate':<28} | {b_sr:19.1f}% | {c_sr:19.1f}% | {format_diff(c_sr, b_sr):<12}")

    # Collision rate
    b_cr = base_metrics['collision_rate'] * 100
    c_cr = champ_metrics['collision_rate'] * 100
    print(f"{'Collision Rate':<28} | {b_cr:19.1f}% | {c_cr:19.1f}% | {format_diff(c_cr, b_cr, invert=True):<12}")

    # Timeout rate
    b_tr = base_metrics['timeout_rate'] * 100
    c_tr = champ_metrics['timeout_rate'] * 100
    print(f"{'Timeout Rate':<28} | {b_tr:19.1f}% | {c_tr:19.1f}% | {format_diff(c_tr, b_tr, invert=True):<12}")

    # Steps to goal
    b_steps = base_fitness if base_fitness is not None else float("inf")
    c_steps = champ_fitness if champ_fitness is not None else float("inf")
    diff_steps = c_steps - b_steps
    step_delta_str = f"{'+' if diff_steps > 0 else ''}{diff_steps:.1f} steps {'★' if diff_steps < 0 else ' '}"
    print(f"{'Avg Steps to Goal (Fitness)':<28} | {b_steps:19.1f}  | {c_steps:19.1f}  | {step_delta_str:<12}")

    # Flight speed
    b_spd = base_metrics['avg_speed']
    c_spd = champ_metrics['avg_speed']
    print(f"{'Average Flight Speed':<28} | {b_spd:15.2f} m/s | {c_spd:15.2f} m/s | {format_diff(c_spd, b_spd, unit=' m/s'):<12}")

    print("-" * 80)
    print("★ indicates desirable performance direction.")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Evaluate and Analyze Genetic Algorithm Results")
    parser.add_argument("--results_dir", type=str, default="ga_results", help="Directory containing evolution_history.json")
    parser.add_argument("--benchmark", action="store_true", help="Run 100-episode head-to-head tournament against base.zip")
    parser.add_argument("--base_model", type=str, default="base.zip", help="Path to base model for benchmarking")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes for head-to-head benchmark (default: 100)")
    parser.add_argument("--visualize", action="store_true", help="Launch Pygame visualizer for the champion model")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist.")
        sys.exit(1)

    try:
        history = load_ga_history(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 1. Print generation-by-generation summary
    print_generation_summary_table(history)

    # 2. Find champion individual
    champion, champion_model_path = find_champion_individual(history, results_dir)
    if champion:
        print("\n" + "*" * 80)
        print(f"                     ALL-TIME EVOLUTIONARY CHAMPION")
        print("*" * 80)
        print(f"Champion ID:       Gen {champion['generation']:02d} | Individual {champion['id']:02d}")
        print(f"Origin Lineage:    {champion.get('origin', 'N/A')}")
        print(f"Fitness (Steps):   {champion['fitness']:.1f} steps to goal")
        m = champion.get("metrics", {})
        print(f"Success Rate:      {m.get('success_rate', 0.0) * 100:.1f}%")
        print(f"Collision Rate:    {m.get('collision_rate', 0.0) * 100:.1f}%")
        print(f"Timeout Rate:      {m.get('timeout_rate', 0.0) * 100:.1f}%")
        print(f"Average Speed:     {m.get('avg_speed', 0.0):.2f} m/s")
        print(f"Model Checkpoint:  {champion_model_path}")
        print("\nChampion Reward Genes:")
        print(json.dumps(champion.get("genes", {}), indent=2))
        print("*" * 80)
    else:
        print("\nNo viable individuals (Success Rate >= 50%) were found across all generations.")

    # 3. Generate evolution graphs
    generate_evolution_plots(history, results_dir)
    try:
        from plot_evolution import (
            plot_individual_existence,
            plot_best_individual_metrics,
            plot_mean_individual_metrics
        )
        plot_individual_existence(history, results_dir / "plot1_individual_existence.png")
        plot_best_individual_metrics(history, results_dir / "plot2_best_individual_metrics.png")
        plot_mean_individual_metrics(history, results_dir / "plot3_mean_individual_metrics.png")
    except Exception as e:
        print(f"[Warning] Could not generate detailed individual plots: {e}")

    # 4. Run head-to-head benchmark if requested
    if args.benchmark and champion:
        benchmark_champion_vs_baseline(
            champion_model_path=champion_model_path,
            champion_genes=champion.get("genes", {}),
            base_model_path=args.base_model,
            num_episodes=args.episodes
        )

    # 5. Launch visualizer if requested
    if args.visualize and champion:
        print(f"\nLaunching Pygame visualizer for champion: {champion_model_path}...")
        os.system(f".venv/bin/python3.12 enjoy.py --model {champion_model_path}")

if __name__ == "__main__":
    main()
