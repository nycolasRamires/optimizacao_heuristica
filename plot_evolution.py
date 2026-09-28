import os
import sys
import json
import base64
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import GENE_BOUNDS from evolution if available, otherwise define default bounds
try:
    from evolution import GENE_BOUNDS
except ImportError:
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

def load_data(history_path: Path) -> dict:
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found at: {history_path}")
    with open(history_path, "r") as f:
        return json.load(f)

def detect_pop_size(data: dict) -> int:
    """Detects the population size dynamically across all recorded generations."""
    sizes = []
    for g in data.get("generations", []):
        s = len(g.get("ranked_survivors", []))
        d = len(g.get("dead_individuals", []))
        sizes.append(s + d)
    return max(sizes) if sizes else 20

# Style definitions for origin markers
ORIGIN_STYLES = {
    "elite": {"color": "#e6ab02", "marker": "*", "size": 280, "label": "Elite (Direct clone)"},
    "tournament": {"color": "#1f78b4", "marker": "s", "size": 180, "label": "Tournament survivor"},
    "tournament_selected": {"color": "#1f78b4", "marker": "s", "size": 180, "label": "Tournament survivor"},
    "rank_selected": {"color": "#1f78b4", "marker": "s", "size": 180, "label": "Rank survivor"},
    "elite_recomb": {"color": "#33a02c", "marker": "D", "size": 180, "label": "Elite Recomb offspring"},
    "random_recomb": {"color": "#984ea3", "marker": "^", "size": 180, "label": "Random Recomb offspring"},
    "random_init": {"color": "#7f7f7f", "marker": "o", "size": 170, "label": "Random Init (Gen 0)"},
    "random_fresh": {"color": "#ff7f00", "marker": "v", "size": 180, "label": "Fresh Immigrant"}
}

# ==============================================================================
# Plot 1A: Standalone Full Population Matrix, Lineage Tracking & Mutations
# ==============================================================================
def plot_individual_lineage_matrix(data: dict, output_path: Path):
    """
    Dedicated high-visibility timeline plot for all individuals across generations.
    - Dynamically scales height with population size (e.g. 20, 25, 30+).
    - Connects each child directly to its parent(s) in the previous generation.
    - Displays the mutation symbol (⚡) directly inside the marker itself.
    """
    generations = data["generations"]
    num_gens = len(generations)
    pop_size = detect_pop_size(data)

    # Dynamic figure dimensions: scales cleanly with pop_size and num_gens
    fig_width = max(16.0, num_gens * 3.6)
    fig_height = max(12.0, pop_size * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=220)

    # Pre-extract all individuals (survivors + dead) per generation with (x, y) coordinates
    gen_individuals = []
    ind_coords_by_gen = []

    for g_idx, g in enumerate(generations):
        survs = g.get("ranked_survivors", [])
        dead = g.get("dead_individuals", [])
        gen_individuals.append({"generation": g["generation"], "survivors": survs, "dead": dead})

        coord_map = {}
        for rank, ind in enumerate(survs):
            coord_map[ind["id"]] = (g_idx, pop_size - rank)
        for d_idx, d_ind in enumerate(dead):
            coord_map[d_ind["id"]] = (g_idx, pop_size - len(survs) - d_idx)
        ind_coords_by_gen.append(coord_map)

    # 1. Draw lineage connection lines from Generation g to Generation g+1
    for g_idx in range(num_gens - 1):
        curr_info = gen_individuals[g_idx]
        next_info = gen_individuals[g_idx + 1]
        prev_coords = ind_coords_by_gen[g_idx]

        curr_survs = curr_info["survivors"]
        next_survs = next_info["survivors"]

        for next_rank, child in enumerate(next_survs):
            child_y = pop_size - next_rank
            child_orig = child.get("origin", "")
            child_genes = np.array([child["genes"][k] for k in sorted(child["genes"].keys())])
            parent_ids = child.get("parent_ids") or []

            # Determine parent coordinates
            parent_y_coords = []

            # Direct lookup by parent_ids
            if parent_ids:
                for pid in parent_ids:
                    if pid in prev_coords:
                        parent_y_coords.append((pid, prev_coords[pid][1]))

            # Fallback for legacy data without parent_ids: match by gene similarity
            if not parent_y_coords and curr_survs:
                matched_curr_rank = None
                for curr_rank, p in enumerate(curr_survs):
                    p_genes = np.array([p["genes"][k] for k in sorted(p["genes"].keys())])
                    if np.allclose(p_genes, child_genes, rtol=1e-4, atol=1e-5):
                        matched_curr_rank = curr_rank
                        break
                if matched_curr_rank is None:
                    # Closest euclidean distance
                    dists = [np.linalg.norm(np.array([p["genes"][k] for k in sorted(p["genes"].keys())]) - child_genes) for p in curr_survs]
                    matched_curr_rank = int(np.argmin(dists))
                parent_y_coords.append((curr_survs[matched_curr_rank]["id"], pop_size - matched_curr_rank))

            # Draw lines to all identified parents (clean lines without badges)
            is_recomb = "recomb" in child_orig or len(parent_y_coords) > 1
            is_clone = "elite" in child_orig or "tournament" in child_orig or "rank" in child_orig

            for pid, p_y in parent_y_coords:
                if is_clone and not child.get("is_mutated", False):
                    # Solid clean line for unmutated clones
                    line_col = "#2b83ba" if "tournament" in child_orig else "#dfc27d"
                    ax.plot(
                        [g_idx, g_idx + 1], [p_y, child_y],
                        color=line_col, linestyle="-", linewidth=1.7, alpha=0.55, zorder=1
                    )
                else:
                    # Dashed line for recombined or mutated offspring
                    line_col = "#e66101" if "elite" in child_orig else "#7570b3"
                    ax.plot(
                        [g_idx, g_idx + 1], [p_y, child_y],
                        color=line_col, linestyle="--", linewidth=1.4, alpha=0.50, zorder=1
                    )

    # 2. Plot all individuals as nodes for each generation
    for g_idx, info in enumerate(gen_individuals):
        survs = info["survivors"]
        dead = info["dead"]

        # Survivors (Ranked 1 to len(survs))
        for rank, ind in enumerate(survs):
            y_pos = pop_size - rank
            orig = ind.get("origin", "random_init")
            style = ORIGIN_STYLES.get(orig, ORIGIN_STYLES["random_init"])
            fit = ind.get("fitness", 0.0)
            sr = ind.get("metrics", {}).get("success_rate", 0.0) * 100.0
            is_mut = ind.get("is_mutated", False) or len(ind.get("mutated_genes", [])) > 0

            # Draw outer symbol
            ax.scatter(
                g_idx, y_pos,
                color=style["color"],
                marker=style["marker"],
                s=style["size"],
                edgecolors="black",
                linewidth=1.0,
                zorder=3
            )

            # Put mutation symbol in the symbol itself
            if is_mut:
                ax.text(
                    g_idx, y_pos, "⚡",
                    fontsize=10.5, ha="center", va="center",
                    color="#ffff00", fontweight="bold",
                    zorder=5
                )

            # Compact informative label next to marker (only when num_gens is small, to prevent 750-text-box collision)
            if num_gens < 12:
                mut_prefix = "⚡ " if is_mut else ""
                label_text = f"{mut_prefix}#{ind['id']:02d} | {fit:.0f}s ({sr:.0f}%)"
                ax.text(
                    g_idx + 0.05, y_pos - 0.12, label_text,
                    fontsize=8.5, color="#1a1a1a", fontweight="bold",
                    verticalalignment="center", zorder=4
                )
            elif rank == 0:
                # Clean callout for generation champion only
                ax.text(
                    g_idx, y_pos + 0.35, f"#{ind['id']:02d} ({fit:.0f}s)",
                    fontsize=8.0, color="#005a32", fontweight="bold",
                    ha="center", va="bottom", zorder=5
                )

        # Dead individuals (Failing dynamic viability threshold)
        for d_idx, d_ind in enumerate(dead):
            y_pos = pop_size - len(survs) - d_idx
            sr = d_ind.get("metrics", {}).get("success_rate", 0.0) * 100.0
            d_id = d_ind.get("id", 0)

            ax.scatter(
                g_idx, y_pos,
                color="#d62728",
                marker="X",
                s=180,
                edgecolors="black",
                linewidth=1.2,
                zorder=3
            )
            if num_gens < 12:
                d_th = d_ind.get("metrics", {}).get("viability_threshold", 0.5) * 100.0
                ax.text(
                    g_idx + 0.05, y_pos - 0.12,
                    f"[DEAD] #{d_id:02d} ({sr:.0f}% < {d_th:.0f}%)",
                    fontsize=8.5, color="#b30000", fontweight="bold",
                    verticalalignment="center", zorder=4
                )

    # Format Axes
    ax.set_yticks(range(1, pop_size + 1))
    ax.set_yticklabels([f"Rank {pop_size - i + 1:02d}" for i in range(1, pop_size + 1)], fontsize=9)
    ax.set_xticks(range(num_gens))
    if num_gens > 15:
        xticklabels = []
        for g_idx, g in enumerate(generations):
            gate = g.get("viability_threshold", 50.0)
            if g_idx in [0, num_gens - 1] or g_idx % 5 == 0:
                xticklabels.append(f"Gen {g['generation']:02d}\n(Gate: {gate:.0f}%)")
            else:
                xticklabels.append(f"G{g['generation']:02d}")
        ax.set_xticklabels(xticklabels, fontsize=9.0, fontweight="bold")
    else:
        ax.set_xticklabels(
            [f"Gen {g['generation']:02d}\n(Gate: {g.get('viability_threshold', 50):.0f}%)" for g in generations],
            fontsize=10.5, fontweight="bold"
        )
    ax.set_xlim(-0.35, num_gens - 0.35)
    ax.set_ylim(0.2, pop_size + 0.8)
    ax.set_xlabel("Generation", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Population Performance Rank (Top = Fastest Time to Goal)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"Plot 1A: Full {pop_size}-Individual Population Matrix, Parent Lineage & In-Symbol Mutation", fontsize=14, fontweight="bold", pad=14)
    ax.grid(True, axis="both", linestyle="--", alpha=0.35)

    # Custom Legend
    legend_handles = [
        plt.Line2D([0], [0], marker=s["marker"], color="w", markerfacecolor=s["color"],
                   markeredgecolor="black", markersize=9, label=s["label"])
        for s in ORIGIN_STYLES.values()
    ]
    legend_handles.append(plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#d62728",
                                     markeredgecolor="black", markersize=10, label="Dead (< Dynamic Viability Gate)"))
    legend_handles.append(plt.Line2D([0], [0], marker="$⚡$", color="black", markerfacecolor="#ffff00",
                                     linestyle="none", markersize=10, label="⚡ Mutation inside symbol"))

    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, -0.06), ncol=4, fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Plot 1A to: {output_path}")

# ==============================================================================
# Plot 1B: Standalone Population Composition & Mortality Stacked Bars
# ==============================================================================
def plot_population_composition(data: dict, output_path: Path):
    generations = data["generations"]
    num_gens = len(generations)
    pop_size = detect_pop_size(data)

    fig, ax = plt.subplots(figsize=(max(11.0, num_gens * 2.2), 8.0), dpi=220)

    origin_counts = defaultdict(lambda: [0] * num_gens)
    dead_counts = [len(g.get("dead_individuals", [])) for g in generations]

    for g_idx, g in enumerate(generations):
        for ind in g.get("ranked_survivors", []):
            orig = ind.get("origin", "random_init")
            origin_counts[orig][g_idx] += 1

    bottoms = np.zeros(num_gens)
    for orig, counts in origin_counts.items():
        style = ORIGIN_STYLES.get(orig, ORIGIN_STYLES["random_init"])
        ax.bar(range(num_gens), counts, bottom=bottoms, color=style["color"],
               edgecolor="black", linewidth=0.7, label=style["label"], alpha=0.85)
        bottoms += np.array(counts)

    # Stack dead count on top with red hatch
    if any(d > 0 for d in dead_counts):
        ax.bar(range(num_gens), dead_counts, bottom=bottoms, color="#d62728", hatch="//",
               edgecolor="black", linewidth=0.8, label="Dead (< Dynamic Viability Gate)", alpha=0.85)
        bottoms += np.array(dead_counts)

    # Annotate total alive and dead on top of bars
    for g_idx in range(num_gens):
        num_s = len(generations[g_idx].get("ranked_survivors", []))
        num_d = dead_counts[g_idx]
        txt = f"{num_s} alive" if num_d == 0 else f"{num_s} alive\n{num_d} DEAD"
        col = "#2ca02c" if num_d == 0 else "#b30000"
        ax.text(g_idx, pop_size + 0.3, txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=col)

    ax.set_xticks(range(num_gens))
    if num_gens > 15:
        xticklabels = [f"Gen {g['generation']:02d}" if (i in [0, num_gens - 1] or i % 5 == 0) else f"G{g['generation']:02d}" for i, g in enumerate(generations)]
        ax.set_xticklabels(xticklabels, fontsize=9.0, fontweight="bold")
    else:
        ax.set_xticklabels([f"Gen {g['generation']:02d}" for g in generations], fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Number of Individuals (Pop Size: {pop_size})", fontsize=11, fontweight="bold")
    ax.set_title(f"Plot 1B: Population Composition & Mortality per Generation (N={pop_size})", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, pop_size * 1.50)
    ax.legend(loc="upper right", fontsize=9.0, framealpha=0.92)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Mortality enumeration text box
    dead_summary_lines = ["Mortality Enumeration (Dynamic Viability Gate):"]
    for g_idx, g in enumerate(generations):
        dead_list = g.get("dead_individuals", [])
        if not dead_list:
            dead_summary_lines.append(f"• Gen {g['generation']:02d}: 0 dead (all {pop_size} survived)")
        else:
            for d in dead_list:
                d_sr = d.get("metrics", {}).get("success_rate", 0.0) * 100.0
                dead_summary_lines.append(f"• Gen {g['generation']:02d}: Ind #{d['id']:02d} DIED (SR={d_sr:.1f}% < 55%)")

    ax.text(
        0.03, 0.96, "\n".join(dead_summary_lines),
        transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fff5f5", ec="#d62728", lw=1.2, alpha=0.95)
    )

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Plot 1B to: {output_path}")

# ==============================================================================
# Plot 1C: Standalone Population Fitness Spread (Boxplot)
# ==============================================================================
def plot_fitness_dispersion(data: dict, output_path: Path):
    generations = data["generations"]
    num_gens = len(generations)

    fig, ax = plt.subplots(figsize=(max(11.0, num_gens * 2.0), 7.5), dpi=220)

    fitness_per_gen = []
    for g in generations:
        survs = g.get("ranked_survivors", [])
        fitness_per_gen.append([s["fitness"] for s in survs if s.get("fitness") is not None])

    ax.boxplot(
        fitness_per_gen,
        positions=range(num_gens),
        patch_artist=True,
        widths=0.45,
        boxprops=dict(facecolor="#c6dbef", color="#1f77b4", linewidth=1.2),
        medianprops=dict(color="#d95f02", linewidth=2.0),
        whiskerprops=dict(color="#1f77b4", linewidth=1.2),
        capprops=dict(color="#1f77b4", linewidth=1.2)
    )

    # Scatter overlay of individual points
    for g_idx, fits in enumerate(fitness_per_gen):
        jitter = np.random.normal(0, 0.04, size=len(fits))
        ax.scatter(g_idx + jitter, fits, color="#386cb0", alpha=0.65, s=30, edgecolors="none")

    # Connect best points
    best_fits = [min(f) for f in fitness_per_gen if f]
    ax.plot(range(num_gens), best_fits, color="#2ca02c", marker="o", linewidth=2.2, label="Best (Fastest) Time")

    ax.set_xticks(range(num_gens))
    ax.set_xticklabels([f"Gen {g['generation']:02d}" for g in generations], fontsize=10, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax.set_ylabel("Timesteps to Goal (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Plot 1C: Population Fitness Dispersion (Median & Spread)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Plot 1C to: {output_path}")

# ==============================================================================
# Plot 2: Best Individual Metrics Across Generations
# ==============================================================================
def plot_best_individual_metrics(data: dict, output_path: Path):
    generations = data["generations"]
    num_gens = len(generations)
    gen_indices = [g["generation"] for g in generations]

    best_fitness = []
    best_sr = []
    best_coll = []
    best_timeout = []
    best_speed = []
    best_return = []
    best_ids = []
    best_origins = []

    for g in generations:
        best = g.get("best_individual")
        if best:
            best_fitness.append(best.get("avg_steps_to_goal", np.nan))
            m = best.get("metrics", {})
            best_sr.append(m.get("success_rate", 0.0) * 100.0)
            best_coll.append(m.get("collision_rate", 0.0) * 100.0)
            best_timeout.append(m.get("timeout_rate", 0.0) * 100.0)
            best_speed.append(m.get("avg_speed", 0.0))
            best_return.append(m.get("avg_return", 0.0))
            best_ids.append(f"Ind {best['id']}")
            best_origins.append(best.get("origin", ""))
        else:
            best_fitness.append(np.nan)
            best_sr.append(0.0)
            best_coll.append(100.0)
            best_timeout.append(0.0)
            best_speed.append(0.0)
            best_return.append(0.0)
            best_ids.append("None")
            best_origins.append("")

    fig, axs = plt.subplots(2, 2, figsize=(16, 11), dpi=220)

    # Determine milestones to prevent text collision across 30 generations
    milestones = set([0, num_gens - 1] + [i for i in gen_indices if i % 5 == 0])
    running_min = float("inf")
    for idx, f in enumerate(best_fitness):
        if not np.isnan(f) and f < running_min:
            running_min = f
            milestones.add(idx)

    # 1. Best Fitness
    ax1 = axs[0, 0]
    ax1.plot(gen_indices, best_fitness, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="Best Fitness")
    ax1.set_title("Plot 2A: Best Individual Fitness (Timesteps to Goal)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Timesteps to Goal (Lower is Better)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    for x, y, ind_id, orig in zip(gen_indices, best_fitness, best_ids, best_origins):
        if x in milestones and not np.isnan(y):
            ax1.annotate(
                f"{y:.1f}s\n({ind_id})",
                xy=(x, y), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#e7f0fa", ec="#1f77b4", alpha=0.9)
            )
    valid_fits = [f for f in best_fitness if not np.isnan(f)]
    if valid_fits:
        y_min, y_max = min(valid_fits), max(valid_fits)
        ax1.set_ylim(max(0, y_min - 20), y_max + 25)

    # 2. Best Individual Outcomes
    ax2 = axs[0, 1]
    ax2.plot(gen_indices, best_sr, marker="^", color="#2ca02c", linewidth=2.2, markersize=8, label="Success Rate (%)")
    ax2.plot(gen_indices, best_coll, marker="x", color="#d62728", linewidth=2.2, markersize=8, label="Collision Rate (%)")
    thresh_schedule = [g.get("viability_threshold", 50.0) for g in generations]
    ax2.plot(gen_indices, thresh_schedule, color="#d62728", linestyle="--", alpha=0.85, linewidth=2.0, label="Dynamic Viability Gate (%)")
    ax2.set_title("Plot 2B: Best Individual Navigation Outcomes (%)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="center right", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    for x, y in zip(gen_indices, best_sr):
        if x in milestones:
            ax2.annotate(f"{y:.0f}%", xy=(x, y), xytext=(0, 6), textcoords="offset points",
                         ha="center", fontsize=8.5, fontweight="bold", color="#1b6e1b")

    # 3. Best Flight Speed
    ax3 = axs[1, 0]
    ax3.plot(gen_indices, best_speed, marker="s", color="#9467bd", linewidth=2.2, markersize=8, label="Average Speed (m/s)")
    ax3.axhline(2.0, color="#d62728", linestyle="--", alpha=0.6, label="Physical Max (2.0 m/s)")
    ax3.axhline(1.2, color="#ff7f0e", linestyle=":", alpha=0.6, label="Safe Threshold (1.2 m/s)")
    ax3.set_title("Plot 2C: Best Individual Average Flight Speed", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Speed (m/s)", fontsize=11, fontweight="bold")
    ax3.set_ylim(0.8, 2.2)
    ax3.legend(loc="lower right", fontsize=9.5)
    ax3.grid(True, linestyle="--", alpha=0.5)

    for x, y in zip(gen_indices, best_speed):
        if x in milestones:
            ax3.annotate(f"{y:.2f} m/s", xy=(x, y), xytext=(0, 7), textcoords="offset points",
                         ha="center", fontsize=8.5, fontweight="bold", color="#5c3566")

    # 4. Cumulative Return
    ax4 = axs[1, 1]
    ax4.plot(gen_indices, best_return, marker="D", color="#17becf", linewidth=2.2, markersize=7, label="Episodic Return")
    ax4.set_title("Plot 2D: Best Individual Average Cumulative Return", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Average Return", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)

    for x, y in zip(gen_indices, best_return):
        if x in milestones:
            ax4.annotate(f"{y:.0f}", xy=(x, y), xytext=(0, 7), textcoords="offset points",
                         ha="center", fontsize=8.5, fontweight="bold", color="#0e6b75")

    # Format xticks cleanly across all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(gen_indices)
        if num_gens > 15:
            ax.set_xticklabels([f"G{i:02d}" if (i not in [0, num_gens - 1] and i % 5 != 0) else f"Gen {i:02d}" for i in gen_indices], fontsize=8.5)
        else:
            ax.set_xticklabels([f"Gen {i:02d}" for i in gen_indices], fontsize=9.5)

    plt.suptitle("EVOLUTIONARY ANALYSIS: BEST INDIVIDUAL METRICS ACROSS GENERATIONS", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Plot 2 to: {output_path}")

# ==============================================================================
# Plot 3: Mean Individual Metrics Across Generations (Population Aggregate)
# ==============================================================================
def plot_mean_individual_metrics(data: dict, output_path: Path):
    generations = data["generations"]
    num_gens = len(generations)
    gen_indices = [g["generation"] for g in generations]

    mean_fitness, std_fitness, min_fitness, max_fitness = [], [], [], []
    mean_sr, std_sr, mean_coll, std_coll, mean_timeout = [], [], [], [], []
    mean_speed, std_speed = [], []
    mean_return, std_return = [], []

    for g in generations:
        survs = g.get("ranked_survivors", [])
        if survs:
            steps = [s["fitness"] for s in survs if s.get("fitness") is not None]
            srs = [s["metrics"]["success_rate"] * 100.0 for s in survs]
            colls = [s["metrics"]["collision_rate"] * 100.0 for s in survs]
            timeouts = [s["metrics"]["timeout_rate"] * 100.0 for s in survs]
            spds = [s["metrics"]["avg_speed"] for s in survs]
            returns = [s["metrics"]["avg_return"] for s in survs]

            mean_fitness.append(np.mean(steps))
            std_fitness.append(np.std(steps))
            min_fitness.append(np.min(steps))
            max_fitness.append(np.max(steps))

            mean_sr.append(np.mean(srs))
            std_sr.append(np.std(srs))
            mean_coll.append(np.mean(colls))
            std_coll.append(np.std(colls))
            mean_timeout.append(np.mean(timeouts))

            mean_speed.append(np.mean(spds))
            std_speed.append(np.std(spds))

            mean_return.append(np.mean(returns))
            std_return.append(np.std(returns))
        else:
            mean_fitness.append(np.nan)
            std_fitness.append(0.0)
            min_fitness.append(np.nan)
            max_fitness.append(np.nan)
            mean_sr.append(0.0)
            std_sr.append(0.0)
            mean_coll.append(100.0)
            std_coll.append(0.0)
            mean_timeout.append(0.0)
            mean_speed.append(0.0)
            std_speed.append(0.0)
            mean_return.append(0.0)
            std_return.append(0.0)

    mean_fitness = np.array(mean_fitness)
    std_fitness = np.array(std_fitness)
    mean_sr = np.array(mean_sr)
    std_sr = np.array(std_sr)
    mean_coll = np.array(mean_coll)
    std_coll = np.array(std_coll)
    mean_speed = np.array(mean_speed)
    std_speed = np.array(std_speed)

    fig, axs = plt.subplots(2, 2, figsize=(16, 11), dpi=220)

    # 1. Population Fitness: Mean ± Std
    ax1 = axs[0, 0]
    ax1.plot(gen_indices, mean_fitness, marker="s", color="#1f77b4", linewidth=2.5, markersize=8, label="Population Mean Steps")
    ax1.fill_between(gen_indices, mean_fitness - std_fitness, mean_fitness + std_fitness, color="#1f77b4", alpha=0.2, label="±1 Std Dev")
    ax1.plot(gen_indices, min_fitness, linestyle="--", color="#2ca02c", linewidth=1.5, label="Generation Best (Min)")
    ax1.plot(gen_indices, max_fitness, linestyle="--", color="#d62728", linewidth=1.5, label="Generation Worst (Max)")
    ax1.set_title("Plot 3A: Population Mean Fitness (Timesteps to Goal)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Timesteps to Goal (Lower is Better)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Navigation Outcomes
    ax2 = axs[0, 1]
    ax2.plot(gen_indices, mean_sr, marker="^", color="#2ca02c", linewidth=2.2, markersize=8, label="Mean Success Rate (%)")
    ax2.fill_between(gen_indices, np.clip(mean_sr - std_sr, 0, 100), np.clip(mean_sr + std_sr, 0, 100), color="#2ca02c", alpha=0.15)
    ax2.plot(gen_indices, mean_coll, marker="x", color="#d62728", linewidth=2.2, markersize=8, label="Mean Collision Rate (%)")
    ax2.fill_between(gen_indices, np.clip(mean_coll - std_coll, 0, 100), np.clip(mean_coll + std_coll, 0, 100), color="#d62728", alpha=0.15)
    ax2.plot(gen_indices, mean_timeout, marker="d", color="#ff7f0e", linewidth=1.8, markersize=7, linestyle=":", label="Mean Timeout Rate (%)")
    thresh_schedule = [g.get("viability_threshold", 50.0) for g in generations]
    ax2.plot(gen_indices, thresh_schedule, color="#d62728", linestyle="--", alpha=0.85, linewidth=2.0, label="Dynamic Viability Gate (%)")
    ax2.set_title("Plot 3B: Population Mean Navigation Outcomes (%)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="center right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Mean Flight Speed
    ax3 = axs[1, 0]
    ax3.plot(gen_indices, mean_speed, marker="s", color="#9467bd", linewidth=2.2, markersize=8, label="Population Mean Speed")
    ax3.fill_between(gen_indices, mean_speed - std_speed, mean_speed + std_speed, color="#9467bd", alpha=0.2, label="±1 Std Dev")
    ax3.axhline(2.0, color="#d62728", linestyle="--", alpha=0.6, label="Physical Max (2.0 m/s)")
    ax3.axhline(1.2, color="#ff7f0e", linestyle=":", alpha=0.6, label="Safe Threshold (1.2 m/s)")
    ax3.set_title("Plot 3C: Population Mean Flight Speed (m/s)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Speed (m/s)", fontsize=11, fontweight="bold")
    ax3.set_ylim(0.8, 2.2)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Elite Gap
    ax4 = axs[1, 1]
    best_steps = [g["best_individual"]["avg_steps_to_goal"] for g in generations if g.get("best_individual")]
    ax4.plot(gen_indices, best_steps, marker="o", color="#2ca02c", linewidth=2.5, markersize=8, label="Best Individual (Fastest)")
    ax4.plot(gen_indices, mean_fitness, marker="s", color="#1f77b4", linewidth=2.2, markersize=8, label="Population Mean")
    ax4.fill_between(gen_indices, best_steps, mean_fitness, color="#2ca02c", alpha=0.15, label="Elite Advantage Margin")
    ax4.set_title("Plot 3D: Elite vs. Population Mean Performance Gap", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Timesteps to Goal (Lower is Better)", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    milestones = set([0, num_gens - 1] + [i for i in gen_indices if i % 5 == 0])
    for x, b, m_val in zip(gen_indices, best_steps, mean_fitness):
        if x in milestones and not (np.isnan(b) or np.isnan(m_val)):
            gap = m_val - b
            ax4.annotate(f"Δ -{gap:.0f}s", xy=(x, (b + m_val) / 2), xytext=(0, 6), textcoords="offset points",
                         ha="center", fontsize=8, fontweight="bold", color="#1b6e1b")

    # Format xticks cleanly across all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(gen_indices)
        if num_gens > 15:
            ax.set_xticklabels([f"G{i:02d}" if (i not in [0, num_gens - 1] and i % 5 != 0) else f"Gen {i:02d}" for i in gen_indices], fontsize=8.5)
        else:
            ax.set_xticklabels([f"Gen {i:02d}" for i in gen_indices], fontsize=9.5)

    plt.suptitle("EVOLUTIONARY ANALYSIS: POPULATION MEAN METRICS & SPREAD", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Plot 3 to: {output_path}")

# ==============================================================================
# Plot 4: Standalone High-Resolution 17-Gene Grid
# ==============================================================================
def plot_gene_evolution(data: dict, output_path: Path):
    """
    Big 17-gene evolution grid with enhanced resolution (300 DPI) for crystal clear inspection.
    """
    generations = data["generations"]
    num_gens = len(generations)
    gen_indices = [g["generation"] for g in generations]
    pop_size = detect_pop_size(data)

    sample_genes = generations[0]["ranked_survivors"][0]["genes"]
    gene_names = list(sample_genes.keys())
    num_genes = len(gene_names)

    # High-resolution 6x3 subplot layout
    fig, axs = plt.subplots(6, 3, figsize=(22, 25), dpi=300)
    axs = axs.flatten()

    drift_stats = []

    for i, gene in enumerate(gene_names):
        ax = axs[i]
        low, high, _ = GENE_BOUNDS.get(gene, (0.0, 1.0, float))

        mean_vals = []
        std_vals = []
        best_vals = []
        all_vals_per_gen = []

        for g in generations:
            all_inds = g.get("ranked_survivors", []) + g.get("dead_individuals", [])
            vals = [ind["genes"][gene] for ind in all_inds if gene in ind.get("genes", {})]
            all_vals_per_gen.append(vals)

            if vals:
                mean_vals.append(float(np.mean(vals)))
                std_vals.append(float(np.std(vals)))
            else:
                mean_vals.append(np.nan)
                std_vals.append(0.0)

            best = g.get("best_individual")
            if best and "genes" in best and gene in best["genes"]:
                best_vals.append(best["genes"][gene])
            else:
                best_vals.append(np.nan)

        mean_vals = np.array(mean_vals)
        std_vals = np.array(std_vals)

        # Plot individual alleles as jittered scatter points
        for g_idx, vals in enumerate(all_vals_per_gen):
            jitter = np.random.normal(0, 0.05, size=len(vals))
            ax.scatter(g_idx + jitter, vals, color="#9ecae1", alpha=0.55, s=20, edgecolors="none", zorder=2)

        # Population Mean and Std Dev Band
        ax.plot(gen_indices, mean_vals, marker="o", color="#08519c", linewidth=2.2, markersize=6.0, label="Pop Mean", zorder=3)
        ax.fill_between(gen_indices, mean_vals - std_vals, mean_vals + std_vals, color="#3182bd", alpha=0.20, zorder=1)

        # Best Individual trajectory
        ax.plot(gen_indices, best_vals, marker="*", color="#238b45", linestyle="--", linewidth=2.0, markersize=8.5, label="Best Ind", zorder=4)

        # Search space bounds
        ax.axhline(low, color="#de2d26", linestyle=":", alpha=0.70, linewidth=1.2)
        ax.axhline(high, color="#de2d26", linestyle=":", alpha=0.70, linewidth=1.2)

        m0 = mean_vals[0] if len(mean_vals) > 0 else 0.0
        m_end = mean_vals[-1] if len(mean_vals) > 0 else 0.0
        pct_change = ((m_end - m0) / (abs(m0) + 1e-6)) * 100.0
        cv_end = (std_vals[-1] / (abs(m_end) + 1e-6)) if m_end != 0 else 0.0
        drift_stats.append({
            "gene": gene,
            "m0": m0,
            "m_end": m_end,
            "pct": pct_change,
            "cv": cv_end
        })

        sign = "+" if pct_change >= 0 else ""
        ax.set_title(f"{gene}\n[{m0:.3g} → {m_end:.3g} ({sign}{pct_change:.1f}%)]", fontsize=10.5, fontweight="bold")
        if num_gens > 15:
            ticks = [j for j in gen_indices if (j in [0, num_gens - 1] or j % 5 == 0)]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"G{j:02d}" for j in ticks], fontsize=8.5)
        else:
            ax.set_xticks(gen_indices)
            ax.set_xticklabels([f"G{j:02d}" for j in gen_indices], fontsize=9)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.45)

        y_margin = (high - low) * 0.08
        ax.set_ylim(low - y_margin, high + y_margin)

    # 18th Panel: Summary Card & Proper Graphical Legend (Side-by-Side)
    ax_summary = axs[num_genes]
    ax_summary.axis("off")

    drift_stats_sorted = sorted(drift_stats, key=lambda x: x["pct"], reverse=True)
    top_increases = drift_stats_sorted[:3]
    top_decreases = sorted(drift_stats, key=lambda x: x["pct"])[:3]
    most_converged = sorted(drift_stats, key=lambda x: x["cv"])[:3]

    summary_text = [
        f"GENE DRIFT SUMMARY (Gen 00-G{gen_indices[-1]:02d})",
        "=" * 34,
        "[+] Top Increases:",
    ]
    for d in top_increases:
        summary_text.append(f"  * {d['gene'][:15]}: +{d['pct']:.1f}%")

    summary_text.append("[-] Top Decreases:")
    for d in top_decreases:
        summary_text.append(f"  * {d['gene'][:15]}: {d['pct']:.1f}%")

    summary_text.append("[*] Most Stabilized (CV):")
    for d in most_converged:
        summary_text.append(f"  * {d['gene'][:15]} ({d['cv']:.2f})")

    ax_summary.text(
        0.02, 0.50, "\n".join(summary_text),
        transform=ax_summary.transAxes, fontsize=8.2, family="monospace",
        verticalalignment="center", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#3182bd", lw=1.2)
    )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Unified graphical legend handles for panel 18
    legend_handles_panel = [
        Line2D([0], [0], color="#238b45", marker="*", linestyle="--", linewidth=2.0, markersize=11, label="Best Ind (Champion)"),
        Line2D([0], [0], color="#08519c", marker="o", linestyle="-", linewidth=2.2, markersize=6.5, label="Population Mean"),
        Patch(facecolor="#3182bd", alpha=0.35, edgecolor="none", label="Spread (±1σ Band)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9ecae1", markeredgecolor="#6baed6", markeredgewidth=0.7, markersize=7.5, alpha=0.85, linestyle="None", label=f"Alleles (N={pop_size})"),
        Line2D([0], [0], color="#de2d26", linestyle=":", linewidth=1.8, label="Bounds [low, high]")
    ]

    # Dedicated graphical legend inside the 18th panel (right-aligned)
    leg_summary = ax_summary.legend(
        handles=legend_handles_panel,
        loc="center right",
        bbox_to_anchor=(0.98, 0.50),
        title="VISUAL LEGEND",
        title_fontsize=9.5,
        fontsize=8.5,
        frameon=True,
        facecolor="#fcfcfc",
        edgecolor="#3182bd",
        fancybox=True,
        shadow=True,
        borderpad=0.7,
        labelspacing=0.6
    )
    leg_summary.get_title().set_fontweight("bold")

    # Global legend handles for top banner
    top_legend_handles = [
        Line2D([0], [0], color="#238b45", marker="*", linestyle="--", linewidth=2.2, markersize=12, label="Best Individual (Gen Champion)"),
        Line2D([0], [0], color="#08519c", marker="o", linestyle="-", linewidth=2.4, markersize=7, label="Population Mean"),
        Patch(facecolor="#3182bd", alpha=0.35, edgecolor="none", label="Population Spread (±1σ Band)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9ecae1", markeredgecolor="#6baed6", markeredgewidth=0.8, markersize=8, alpha=0.85, linestyle="None", label=f"Individual Alleles (N={pop_size})"),
        Line2D([0], [0], color="#de2d26", linestyle=":", linewidth=2.0, label="Search Space Bounds [low, high]")
    ]

    plt.suptitle("EVOLUTIONARY ANALYSIS: GENE ALLELE DRIFT & CONVERGENCE ACROSS 17 HYPERPARAMETERS", fontsize=17, fontweight="bold", y=0.995)

    # Top Global Banner Legend spanning all columns
    fig.legend(
        handles=top_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.983),
        ncol=5,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#90a4ae",
        fontsize=12.0,
        fancybox=True,
        shadow=True,
        borderpad=0.6,
        columnspacing=2.0
    )

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Plot 4 (High Resolution 300 DPI) to: {output_path}")

    # Automatically refresh HTML dashboard
    out_dir = output_path.parent
    img_paths = {
        "plot1a": out_dir / "plot1a_individual_lineage.png",
        "plot1b": out_dir / "plot1b_population_composition.png",
        "plot1c": out_dir / "plot1c_fitness_dispersion.png",
        "plot2": out_dir / "plot2_best_individual_metrics.png",
        "plot3": out_dir / "plot3_mean_individual_metrics.png",
        "plot4": output_path
    }
    try:
        generate_html_dashboard(data, out_dir, img_paths)
    except Exception as e:
        print(f"[HTML Dashboard Warning] {e}")

# ==============================================================================
# Standalone Embedded HTML Dashboard Generator
# ==============================================================================
def generate_html_dashboard(data: dict, output_dir: Path, image_paths: dict) -> Path:
    """
    Generates a single self-contained HTML dashboard with all plots embedded as Base64.
    Includes download buttons for every high-resolution image and executive summaries.
    """
    generations = data.get("generations", [])
    num_gens = len(generations)
    pop_size = detect_pop_size(data)

    latest_gen = generations[-1] if generations else {}
    last_best = latest_gen.get("best_individual", {})
    last_survs = len(latest_gen.get("ranked_survivors", []))
    last_dead = len(latest_gen.get("dead_individuals", []))

    # Find All-Time Champion
    all_time_champ = None
    champ_gen = None
    for g in generations:
        for s in g.get("ranked_survivors", []):
            if all_time_champ is None or s["fitness"] < all_time_champ["fitness"]:
                all_time_champ = s
                champ_gen = g["generation"]

    champ_id = all_time_champ["id"] if all_time_champ else "N/A"
    champ_fitness = f"{all_time_champ['fitness']:.1f} steps" if all_time_champ else "N/A"
    champ_sr = f"{all_time_champ.get('metrics', {}).get('success_rate', 0.0)*100:.1f}%" if all_time_champ else "N/A"
    champ_speed = f"{all_time_champ.get('metrics', {}).get('avg_speed', 0.0):.2f} m/s" if all_time_champ else "N/A"

    # Encode images to base64
    b64_images = {}
    for name, p in image_paths.items():
        if p.exists():
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                b64_images[name] = f"data:image/png;base64,{b64}"
        else:
            b64_images[name] = ""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniDrone 2D - Evolutionary RL Optimization Dashboard</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-orange: #fb923c;
            --accent-red: #f87171;
            --border-color: #334155;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 24px 16px;
        }}
        .container {{
            max-width: 1440px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 28px 36px;
            margin-bottom: 28px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        }}
        h1 {{
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.05rem;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 20px;
        }}
        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        .metric-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 4px;
        }}
        .metric-sub {{
            font-size: 0.82rem;
            color: var(--accent-green);
            margin-top: 2px;
        }}
        .nav-pills {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 28px;
            padding: 8px;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .nav-pill {{
            color: var(--text-secondary);
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        .nav-pill:hover {{
            color: var(--text-primary);
            background: rgba(56, 189, 248, 0.15);
        }}
        .plot-section {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 32px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}
        .plot-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }}
        .plot-title {{
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-primary);
        }}
        .plot-desc {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 4px;
        }}
        .btn-download {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #2563eb;
            color: #ffffff;
            text-decoration: none;
            font-size: 0.88rem;
            font-weight: 600;
            padding: 10px 18px;
            border-radius: 8px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
        }}
        .btn-download:hover {{
            background: #1d4ed8;
            transform: translateY(-1px);
        }}
        .plot-img-container {{
            width: 100%;
            overflow-x: auto;
            text-align: center;
            padding: 8px 0;
            background: #0b1120;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .plot-img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            display: inline-block;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }}
        footer {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
            padding: 24px 0 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>OmniDrone 2D — Evolutionary RL Optimization Dashboard</h1>
            <p class="subtitle">Continuous Reward & Penalty Tuning across {num_gens} Generations (N={pop_size} Individuals/Gen)</p>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Completed Generations</div>
                    <div class="metric-value">{num_gens}</div>
                    <div class="metric-sub">Generations 00 to {generations[-1]['generation'] if generations else 0:02d}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Population Size</div>
                    <div class="metric-value">{pop_size}</div>
                    <div class="metric-sub">{last_survs} Alive | {last_dead} Dead</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">All-Time Champion</div>
                    <div class="metric-value">{champ_fitness}</div>
                    <div class="metric-sub">Gen {champ_gen:02d} | Ind #{champ_id} (SR: {champ_sr})</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Viability Gate</div>
                    <div class="metric-value">55.0%</div>
                    <div class="metric-sub">Dies if Success Rate &lt; 55%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Current Best Flight Speed</div>
                    <div class="metric-value">{champ_speed}</div>
                    <div class="metric-sub">Max Velocity Limit: 2.0 m/s</div>
                </div>
            </div>
        </header>

        <nav class="nav-pills">
            <a href="#plot1a" class="nav-pill">Plot 1A: Population Matrix & Lineages</a>
            <a href="#plot1b" class="nav-pill">Plot 1B: Composition & Mortality</a>
            <a href="#plot1c" class="nav-pill">Plot 1C: Fitness Dispersion</a>
            <a href="#plot2" class="nav-pill">Plot 2: Best Individual Metrics</a>
            <a href="#plot3" class="nav-pill">Plot 3: Population Mean & Spread</a>
            <a href="#plot4" class="nav-pill">Plot 4: 17-Gene Evolution Grid</a>
        </nav>

        <!-- Plot 1A -->
        <section id="plot1a" class="plot-section">
            <div class="plot-header">
                <div>
                    <h2 class="plot-title">Plot 1A: Full Population Matrix, Lineage Continuity & In-Symbol Mutations</h2>
                    <p class="plot-desc">Tracking all {pop_size} individuals per generation. Direct connection to parent(s) with mutation symbol (⚡) displayed inside the candidate icon.</p>
                </div>
                <a href="{b64_images.get('plot1a', '#')}" download="plot1a_lineage_matrix.png" class="btn-download">
                    ⬇ Download High-Res PNG
                </a>
            </div>
            <div class="plot-img-container">
                <img src="{b64_images.get('plot1a', '')}" alt="Plot 1A Lineage Matrix" class="plot-img">
            </div>
        </section>

        <!-- Plot 1B -->
        <section id="plot1b" class="plot-section">
            <div class="plot-header">
                <div>
                    <h2 class="plot-title">Plot 1B: Population Composition & Mortality per Generation</h2>
                    <p class="plot-desc">Stacked origin composition (Elites, Recombinant Offspring, Tournament Survivors) and enumeration of non-viable individuals (&lt; 55% SR).</p>
                </div>
                <a href="{b64_images.get('plot1b', '#')}" download="plot1b_population_composition.png" class="btn-download">
                    ⬇ Download High-Res PNG
                </a>
            </div>
            <div class="plot-img-container">
                <img src="{b64_images.get('plot1b', '')}" alt="Plot 1B Composition" class="plot-img">
            </div>
        </section>

        <!-- Plot 1C -->
        <section id="plot1c" class="plot-section">
            <div class="plot-header">
                <div>
                    <h2 class="plot-title">Plot 1C: Population Fitness Dispersion</h2>
                    <p class="plot-desc">Distribution of timesteps-to-goal among survivors across generations with individual scatter overlay and champion convergence line.</p>
                </div>
                <a href="{b64_images.get('plot1c', '#')}" download="plot1c_fitness_dispersion.png" class="btn-download">
                    ⬇ Download High-Res PNG
                </a>
            </div>
            <div class="plot-img-container">
                <img src="{b64_images.get('plot1c', '')}" alt="Plot 1C Dispersion" class="plot-img">
            </div>
        </section>

        <!-- Plot 2 -->
        <section id="plot2" class="plot-section">
            <div class="plot-header">
                <div>
                    <h2 class="plot-title">Plot 2: Best Individual Performance Metrics</h2>
                    <p class="plot-desc">Tracking the generation champion's timesteps to goal, navigation outcomes (Success, Collision, Timeout rates), flight speed, and cumulative return.</p>
                </div>
                <a href="{b64_images.get('plot2', '#')}" download="plot2_best_individual_metrics.png" class="btn-download">
                    ⬇ Download High-Res PNG
                </a>
            </div>
            <div class="plot-img-container">
                <img src="{b64_images.get('plot2', '')}" alt="Plot 2 Best Metrics" class="plot-img">
            </div>
        </section>

        <!-- Plot 3 -->
        <section id="plot3" class="plot-section">
            <div class="plot-header">
                <div>
                    <h2 class="plot-title">Plot 3: Population Mean Metrics & Spread</h2>
                    <p class="plot-desc">Tracking population-wide averages with ±1σ standard deviation envelopes, outcome rates, and the selective advantage margin (Δ steps).</p>
                </div>
                <a href="{b64_images.get('plot3', '#')}" download="plot3_mean_individual_metrics.png" class="btn-download">
                    ⬇ Download High-Res PNG
                </a>
            </div>
            <div class="plot-img-container">
                <img src="{b64_images.get('plot3', '')}" alt="Plot 3 Mean Metrics" class="plot-img">
            </div>
        </section>

        <!-- Plot 4 -->
        <section id="plot4" class="plot-section">
            <div class="plot-header">
                <div>
                    <h2 class="plot-title">Plot 4: 17-Gene Allele Drift & Convergence Grid (High Resolution 300 DPI)</h2>
                    <p class="plot-desc">Comprehensive grid tracking trajectories, population mean, individual alleles (N={pop_size}), and bounds for all 17 reward/penalty parameters.</p>
                </div>
                <a href="{b64_images.get('plot4', '#')}" download="plot4_gene_evolution.png" class="btn-download">
                    ⬇ Download High-Res PNG
                </a>
            </div>
            <div class="plot-img-container">
                <img src="{b64_images.get('plot4', '')}" alt="Plot 4 Gene Grid" class="plot-img">
            </div>
        </section>

        <footer>
            OmniDrone 2D Simulation Engine &middot; Autonomous Reinforcement Learning &middot; Self-Contained Offline Report
        </footer>
    </div>
</body>
</html>
"""
    html_path = output_dir / "evolution_dashboard.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✓ Generated self-contained HTML Dashboard: {html_path}")
    return html_path

# ==============================================================================
# Legacy Compatibility & Master Plot Generator
# ==============================================================================
def plot_individual_existence(data: dict, output_path: Path):
    """
    Maintains compatibility with evolution.py while generating separate 1A, 1B, 1C plots.
    """
    out_dir = output_path.parent
    p1a = out_dir / "plot1a_individual_lineage.png"
    p1b = out_dir / "plot1b_population_composition.png"
    p1c = out_dir / "plot1c_fitness_dispersion.png"

    plot_individual_lineage_matrix(data, p1a)
    plot_population_composition(data, p1b)
    plot_fitness_dispersion(data, p1c)

    # Save copy to output_path for backward compatibility
    if output_path != p1a:
        import shutil
        shutil.copyfile(p1a, output_path)

def generate_all_plots(history_path: Path, output_dir: Path):
    data = load_data(history_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    p1a = output_dir / "plot1a_individual_lineage.png"
    p1b = output_dir / "plot1b_population_composition.png"
    p1c = output_dir / "plot1c_fitness_dispersion.png"
    p1_legacy = output_dir / "plot1_individual_existence.png"
    p2 = output_dir / "plot2_best_individual_metrics.png"
    p3 = output_dir / "plot3_mean_individual_metrics.png"
    p4 = output_dir / "plot4_gene_evolution.png"

    plot_individual_lineage_matrix(data, p1a)
    plot_population_composition(data, p1b)
    plot_fitness_dispersion(data, p1c)
    
    # Save legacy duplicate
    import shutil
    shutil.copyfile(p1a, p1_legacy)

    plot_best_individual_metrics(data, p2)
    plot_mean_individual_metrics(data, p3)
    plot_gene_evolution(data, p4)

    image_paths = {
        "plot1a": p1a,
        "plot1b": p1b,
        "plot1c": p1c,
        "plot2": p2,
        "plot3": p3,
        "plot4": p4
    }

    html_path = generate_html_dashboard(data, output_dir, image_paths)
    return html_path

def main():
    parser = argparse.ArgumentParser(description="Generate GA Plots & Embedded HTML Dashboard")
    parser.add_argument("--history", type=str, default="ga_results/evolution_history.json", help="Path to evolution_history.json")
    parser.add_argument("--output_dir", type=str, default="ga_results", help="Directory to save generated plots and HTML")
    args = parser.parse_args()

    history_path = Path(args.history)
    output_dir = Path(args.output_dir)

    html_file = generate_all_plots(history_path, output_dir)
    print(f"\nAll plots and standalone HTML dashboard successfully generated in: {output_dir.resolve()}")
    print(f"Open dashboard: file://{html_file.resolve()}")

if __name__ == "__main__":
    main()
