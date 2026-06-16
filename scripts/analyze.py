import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RESULTS = "results/phase1_results.csv"
FIGURES = "results/figures"
os.makedirs(FIGURES, exist_ok=True)

df = pd.read_csv(RESULTS)
# only successful trials for metric analysis
ok = df[df["planning_success"] == True].copy()

SUB_PHASES = ["sparse", "moderate", "dense"]
ALGORITHMS = ["a_star", "min_jerk", "RRT", "RRT-Connect", "RRT*", "PRM", "APF", "CHOMP", "STOMP"]
ALGO_COLORS = {
    "a_star": "#2ecc71", "min_jerk": "#e74c3c",
    "RRT": "#3498db", "RRT-Connect": "#1abc9c", "RRT*": "#9b59b6", "PRM": "#f39c12",
    "APF": "#e67e22", "CHOMP": "#1f77b4", "STOMP": "#d35400"
}

sns.set_theme(style="whitegrid", font_scale=1.1)

# ============================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================
print("=" * 80)
print("  DESCRIPTIVE STATISTICS")
print("=" * 80)

metrics = ["path_smoothness_msj", "path_planning_time", "path_execution_time", "path_optimality_ratio"]
desc_rows = []

for sp in SUB_PHASES:
    for algo in ALGORITHMS:
        sub = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == algo)]
        if len(sub) == 0:
            continue
        row = {"sub_phase": sp, "algorithm": algo, "n_success": len(sub)}
        for m in metrics:
            vals = sub[m].dropna()
            if len(vals) == 0:
                continue
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_std"] = vals.std()
            row[f"{m}_median"] = vals.median()
            row[f"{m}_min"] = vals.min()
            row[f"{m}_max"] = vals.max()
            row[f"{m}_p25"] = vals.quantile(0.25)
            row[f"{m}_p75"] = vals.quantile(0.75)
        # collision rate
        col = sub["collision_flag"]
        row["collision_rate"] = col.sum() / len(col) * 100
        desc_rows.append(row)

desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv("results/descriptive_stats.csv", index=False)
print("  saved results/descriptive_stats.csv")

# ============================================================
# 2. BOX PLOTS — one per metric
# ============================================================
print("\n  generating box plots...")

for m, label, log_scale in [
    ("path_smoothness_msj", "MSJ (rad²/s⁵)", True),
    ("path_planning_time", "Planning Time (s)", True),
    ("path_execution_time", "Execution Time (s)", False),
    ("path_optimality_ratio", "Path Optimality Ratio", False),
]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, sp in enumerate(SUB_PHASES):
        sub = ok[ok["sub_phase"] == sp].dropna(subset=[m])
        if len(sub) == 0:
            continue
        algos_present = [a for a in ALGORITHMS if a in sub["algorithm"].values]
        colors = [ALGO_COLORS.get(a, "#999") for a in algos_present]
        sns.boxplot(data=sub[sub["algorithm"].isin(algos_present)],
                    x="algorithm", y=m, order=algos_present,
                    palette=colors, ax=axes[i])
        axes[i].set_title(sp.upper())
        axes[i].set_xlabel("")
        axes[i].tick_params(axis='x', rotation=45)
        if log_scale and sub[m].min() > 0:
            axes[i].set_yscale("log")
    axes[0].set_ylabel(label)
    plt.tight_layout()
    fname = f"{FIGURES}/boxplot_{m}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"    {fname}")

# ============================================================
# 3. COLLISION RATE BAR CHART
# ============================================================
print("\n  generating collision rate chart...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for i, sp in enumerate(SUB_PHASES):
    rates = []
    algos_present = []
    for algo in ALGORITHMS:
        sub = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == algo)]
        if len(sub) == 0:
            continue
        rates.append(sub["collision_flag"].sum() / len(sub) * 100)
        algos_present.append(algo)
    colors = [ALGO_COLORS.get(a, "#999") for a in algos_present]
    axes[i].bar(range(len(algos_present)), rates, color=colors)
    axes[i].set_xticks(range(len(algos_present)))
    axes[i].set_xticklabels(algos_present, rotation=45, ha="right")
    axes[i].set_title(sp.upper())
    axes[i].set_ylim(0, 105)
axes[0].set_ylabel("Collision Rate (%)")
plt.tight_layout()
fname = f"{FIGURES}/collision_rate.png"
plt.savefig(fname, dpi=150)
plt.close()
print(f"    {fname}")

# ============================================================
# 4. SUCCESS RATE BAR CHART
# ============================================================
print("\n  generating success rate chart...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for i, sp in enumerate(SUB_PHASES):
    rates = []
    algos_present = []
    for algo in ALGORITHMS:
        sub = df[(df["sub_phase"] == sp) & (df["algorithm"] == algo)]
        if len(sub) == 0:
            continue
        rates.append(sub["planning_success"].sum() / len(sub) * 100)
        algos_present.append(algo)
    colors = [ALGO_COLORS.get(a, "#999") for a in algos_present]
    axes[i].bar(range(len(algos_present)), rates, color=colors)
    axes[i].set_xticks(range(len(algos_present)))
    axes[i].set_xticklabels(algos_present, rotation=45, ha="right")
    axes[i].set_title(sp.upper())
    axes[i].set_ylim(0, 105)
axes[0].set_ylabel("Planning Success Rate (%)")
plt.tight_layout()
fname = f"{FIGURES}/success_rate.png"
plt.savefig(fname, dpi=150)
plt.close()
print(f"    {fname}")

# ============================================================
# 5. PLANNING TIME VS COLLISION TRADE-OFF SCATTER
# ============================================================
print("\n  generating trade-off scatter...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, sp in enumerate(SUB_PHASES):
    for algo in ALGORITHMS:
        sub = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == algo)]
        if len(sub) == 0:
            continue
        pt = sub["path_planning_time"].mean()
        col = sub["collision_flag"].sum() / len(sub) * 100
        axes[i].scatter(pt, col, s=120, c=ALGO_COLORS.get(algo, "#999"),
                        label=algo, zorder=5, edgecolors="black", linewidths=0.5)
    axes[i].set_title(sp.upper())
    axes[i].set_xlabel("Mean Planning Time (s)")
    axes[i].set_ylim(-5, 105)
    if i == 0:
        axes[i].set_ylabel("Collision Rate (%)")
    if i == 2:
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
fname = f"{FIGURES}/tradeoff_time_vs_collision.png"
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"    {fname}")

# ============================================================
# 6. RADAR PLOTS PER ALGORITHM
# ============================================================
print("\n  generating radar plots...")

def normalize_metric(values, lower_better=True):
    mn, mx = values.min(), values.max()
    if mx - mn < 1e-12:
        return np.ones_like(values) * 0.5
    normed = (values - mn) / (mx - mn)
    return 1 - normed if lower_better else normed

radar_metrics = ["path_planning_time", "path_smoothness_msj", "path_optimality_ratio",
                 "collision_rate", "path_execution_time"]
radar_labels = ["Plan Speed", "Smoothness", "Path Quality", "Safety", "Exec Speed"]

for sp in SUB_PHASES:
    sp_data = desc_df[desc_df["sub_phase"] == sp].copy()
    if len(sp_data) < 2:
        continue
    values = {}
    for _, row in sp_data.iterrows():
        algo = row["algorithm"]
        values[algo] = [
            row.get("path_planning_time_mean", 0),
            row.get("path_smoothness_msj_mean", 0),
            row.get("path_optimality_ratio_mean", 1),
            row.get("collision_rate", 0),
            row.get("path_execution_time_mean", 0),
        ]
    val_df = pd.DataFrame(values, index=radar_labels).T
    for col in val_df.columns:
        val_df[col] = normalize_metric(val_df[col], lower_better=True)

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for algo in val_df.index:
        vals = val_df.loc[algo].tolist() + [val_df.loc[algo].iloc[0]]
        ax.plot(angles, vals, linewidth=2, label=algo,
                color=ALGO_COLORS.get(algo, "#999"))
        ax.fill(angles, vals, alpha=0.05, color=ALGO_COLORS.get(algo, "#999"))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)
    ax.set_title(f"Algorithm Comparison — {sp.upper()}", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    fname = f"{FIGURES}/radar_{sp}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    {fname}")

# ============================================================
# 7. ANOVA / KRUSKAL-WALLIS
# ============================================================
print("\n  running statistical tests...")

stat_rows = []
for sp in SUB_PHASES:
    for m in metrics:
        groups = []
        group_names = []
        for algo in ALGORITHMS:
            sub = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == algo)]
            vals = sub[m].dropna()
            if len(vals) >= 3:
                groups.append(vals.values)
                group_names.append(algo)
        if len(groups) < 2:
            continue
        # normality check
        normal = all(stats.shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)
        if normal and len(groups) >= 2:
            stat, pval = stats.f_oneway(*groups)
            test = "ANOVA"
        else:
            stat, pval = stats.kruskal(*groups)
            test = "Kruskal-Wallis"
        stat_rows.append({"sub_phase": sp, "metric": m, "test": test,
                          "statistic": stat, "p_value": pval,
                          "significant": pval < 0.05, "n_groups": len(groups)})
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"    {sp:>8} {m:>30} {test:>15} p={pval:.4f} {sig}")

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv("results/statistical_tests.csv", index=False)
print("  saved results/statistical_tests.csv")

# ============================================================
# DONE
# ============================================================
print(f"\n{'='*80}")
print(f"  ANALYSIS COMPLETE")
print(f"  descriptive stats: results/descriptive_stats.csv")
print(f"  statistical tests: results/statistical_tests.csv")
print(f"  figures: {FIGURES}/")
for f in sorted(os.listdir(FIGURES)):
    print(f"    {f}")
print(f"{'='*80}")
