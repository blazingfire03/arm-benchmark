import os, json
import numpy as np
import pandas as pd

RESULTS = "results/phase1_results.csv"
OUTPUT = "results/phase1_summary.json"

df = pd.read_csv(RESULTS)
ok = df[df["planning_success"] == True].copy()

SUB_PHASES = ["sparse", "moderate", "dense"]
ALGORITHMS = ["a_star", "min_jerk", "RRT", "RRT-Connect", "RRT*", "PRM", "APF", "CHOMP", "STOMP"]
ROLES = {"a_star": "ceiling_baseline", "min_jerk": "floor_baseline",
         "RRT": "sampling_based", "RRT-Connect": "sampling_based",
         "RRT*": "sampling_based", "PRM": "roadmap",
         "APF": "reactive", "CHOMP": "optimization", "STOMP": "optimization"}

summary = {
    "experiment": {
        "name": "Phase 1: Fixed Obstacle Evaluation",
        "robot": "Franka Emika Panda 7-DOF",
        "simulator": "PyBullet",
        "sub_phases": {
            "sparse": {"obstacle_count": 4, "target_collision_rate": "<=20%"},
            "moderate": {"obstacle_count": 12, "target_collision_rate": "50-70%"},
            "dense": {"obstacle_count": 16, "target_collision_rate": ">=95%"},
        },
        "trials_per_algorithm_per_subphase": int(len(df) / (len(ALGORITHMS) * len(SUB_PHASES))),
        "total_trials": len(df),
        "metrics": {
            "MSJ": "Mean Squared Jerk (rad²/s⁵) — lower is smoother, min-jerk is optimal",
            "planning_time": "Wall-clock time to compute path (s) — lower is faster",
            "execution_time": "Time to traverse planned path (s) — lower is shorter path",
            "POR": "Path Optimality Ratio (d_path / d_optimal) — 1.0 = A* optimal",
            "collision_rate": "Percentage of trials with arm-obstacle collision — 0% is ideal",
            "success_rate": "Percentage of trials where planning succeeded",
        },
    },
    "baselines": {},
    "algorithms": {},
    "per_subphase": {},
    "pass_fail": {},
    "rankings": {},
}

# helper
def stats(vals):
    if len(vals) == 0:
        return None
    return {
        "mean": round(float(vals.mean()), 6),
        "std": round(float(vals.std()), 6),
        "median": round(float(vals.median()), 6),
        "min": round(float(vals.min()), 6),
        "max": round(float(vals.max()), 6),
        "p25": round(float(vals.quantile(0.25)), 6),
        "p75": round(float(vals.quantile(0.75)), 6),
        "count": int(len(vals)),
    }

# global min-jerk MSJ for ratio computation
mj_msj_global = ok[ok["algorithm"] == "min_jerk"]["path_smoothness_msj"].dropna().mean()

# per algorithm, per sub-phase
for sp in SUB_PHASES:
    summary["per_subphase"][sp] = {}

    for algo in ALGORITHMS:
        sub_all = df[(df["sub_phase"] == sp) & (df["algorithm"] == algo)]
        sub_ok = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == algo)]

        if len(sub_all) == 0:
            continue

        n_total = len(sub_all)
        n_success = int(sub_all["planning_success"].sum())
        n_collision = int(sub_ok["collision_flag"].sum()) if len(sub_ok) > 0 else 0
        col_rate = round(n_collision / max(n_success, 1) * 100, 1)
        success_rate = round(n_success / n_total * 100, 1)

        msj = sub_ok["path_smoothness_msj"].dropna()
        pt = sub_all["path_planning_time"].dropna()
        et = sub_ok["path_execution_time"].dropna()
        por = sub_ok["path_optimality_ratio"].dropna()

        msj_ratio = round(float(msj.mean() / mj_msj_global), 1) if len(msj) > 0 and mj_msj_global > 0 else None

        entry = {
            "algorithm": algo,
            "role": ROLES.get(algo, "unknown"),
            "sub_phase": sp,
            "trials": n_total,
            "success_count": n_success,
            "success_rate_pct": success_rate,
            "collision_count": n_collision,
            "collision_rate_pct": col_rate,
            "msj_ratio_vs_minjerk": msj_ratio,
            "metrics": {
                "path_smoothness_msj": stats(msj),
                "path_planning_time": stats(pt),
                "path_execution_time": stats(et),
                "path_optimality_ratio": stats(por),
            },
        }

        summary["per_subphase"][sp][algo] = entry

# overall per algorithm (across all sub-phases)
for algo in ALGORITHMS:
    sub_all = df[df["algorithm"] == algo]
    sub_ok = ok[ok["algorithm"] == algo]

    if len(sub_all) == 0:
        continue

    n_total = len(sub_all)
    n_success = int(sub_all["planning_success"].sum())
    n_collision = int(sub_ok["collision_flag"].sum()) if len(sub_ok) > 0 else 0

    msj = sub_ok["path_smoothness_msj"].dropna()
    pt = sub_all["path_planning_time"].dropna()
    et = sub_ok["path_execution_time"].dropna()
    por = sub_ok["path_optimality_ratio"].dropna()

    msj_ratio = round(float(msj.mean() / mj_msj_global), 1) if len(msj) > 0 and mj_msj_global > 0 else None

    entry = {
        "algorithm": algo,
        "role": ROLES.get(algo, "unknown"),
        "total_trials": n_total,
        "total_success": n_success,
        "overall_success_rate_pct": round(n_success / n_total * 100, 1),
        "overall_collision_rate_pct": round(n_collision / max(n_success, 1) * 100, 1),
        "msj_ratio_vs_minjerk": msj_ratio,
        "overall_metrics": {
            "path_smoothness_msj": stats(msj),
            "path_planning_time": stats(pt),
            "path_execution_time": stats(et),
            "path_optimality_ratio": stats(por),
        },
        "per_subphase_summary": {},
    }

    for sp in SUB_PHASES:
        sp_all = sub_all[sub_all["sub_phase"] == sp]
        sp_ok = sub_ok[sub_ok["sub_phase"] == sp]
        if len(sp_all) == 0:
            continue
        sp_col = int(sp_ok["collision_flag"].sum()) if len(sp_ok) > 0 else 0
        entry["per_subphase_summary"][sp] = {
            "success_rate_pct": round(int(sp_all["planning_success"].sum()) / len(sp_all) * 100, 1),
            "collision_rate_pct": round(sp_col / max(len(sp_ok), 1) * 100, 1),
            "mean_planning_time": round(float(sp_all["path_planning_time"].mean()), 4),
            "mean_msj": round(float(sp_ok["path_smoothness_msj"].dropna().mean()), 2) if len(sp_ok["path_smoothness_msj"].dropna()) > 0 else None,
        }

    dest = "baselines" if algo in ("a_star", "min_jerk") else "algorithms"
    summary[dest][algo] = entry

# pass/fail evaluation
for sp in SUB_PHASES:
    sp_df = df[df["sub_phase"] == sp]
    mj_col = sp_df[sp_df["algorithm"] == "min_jerk"]["collision_flag"].sum()
    mj_col_rate = mj_col / len(sp_df[sp_df["algorithm"] == "min_jerk"]) * 100
    as_pt = sp_df[sp_df["algorithm"] == "a_star"]["path_planning_time"].mean()

    summary["pass_fail"][sp] = {
        "floor_collision_rate": round(float(mj_col_rate), 1),
        "ceiling_plan_time": round(float(as_pt), 4),
        "results": {},
    }

    test_algos = [a for a in ALGORITHMS if a not in ("a_star", "min_jerk")]
    for algo in test_algos:
        sub = sp_df[sp_df["algorithm"] == algo]
        sub_ok = sub[sub["planning_success"] == True]
        if len(sub_ok) == 0:
            summary["pass_fail"][sp]["results"][algo] = {
                "collision_pass": None, "speed_pass": None,
                "success_pct": 0, "overall": "FAIL", "reason": "no successful trials"
            }
            continue

        col = sub_ok["collision_flag"].sum() / len(sub_ok) * 100
        pt = sub["path_planning_time"].mean()
        success_pct = len(sub_ok) / len(sub) * 100

        col_pass = col < mj_col_rate or mj_col_rate == 0
        pt_pass = pt < as_pt
        overall = col_pass and pt_pass and success_pct >= 80

        summary["pass_fail"][sp]["results"][algo] = {
            "collision_rate_pct": round(float(col), 1),
            "collision_pass": col_pass,
            "planning_time_s": round(float(pt), 4),
            "speed_pass": pt_pass,
            "success_pct": round(float(success_pct), 1),
            "overall": "PASS" if overall else "FAIL",
            "fail_reasons": (
                [] if overall else
                ([f"collision {col:.1f}% >= floor {mj_col_rate:.1f}%"] if not col_pass else []) +
                ([f"plan time {pt:.2f}s >= ceiling {as_pt:.2f}s"] if not pt_pass else []) +
                ([f"success rate {success_pct:.0f}% < 80%"] if success_pct < 80 else [])
            ),
        }

# rankings per sub-phase
for sp in SUB_PHASES:
    test_algos = [a for a in ALGORITHMS if a not in ("a_star", "min_jerk")]
    sp_ok = ok[ok["sub_phase"] == sp]

    rankings = {}
    for metric, ascending in [
        ("collision_rate", True), ("path_planning_time", True),
        ("path_smoothness_msj", True), ("path_execution_time", True)
    ]:
        scores = []
        for algo in test_algos:
            sub = sp_ok[sp_ok["algorithm"] == algo]
            if len(sub) == 0:
                continue
            if metric == "collision_rate":
                val = sub["collision_flag"].sum() / len(sub) * 100
            else:
                vals = sub[metric].dropna()
                val = float(vals.mean()) if len(vals) > 0 else float('inf')
            scores.append({"algorithm": algo, "value": round(val, 4)})

        scores.sort(key=lambda x: x["value"], reverse=not ascending)
        rankings[metric] = [{"rank": i+1, **s} for i, s in enumerate(scores)]

    summary["rankings"][sp] = rankings

with open(OUTPUT, "w") as f:
    json.dump(summary, f, indent=2)

print(f"saved: {OUTPUT}")
print(f"  {len(summary['baselines'])} baselines")
print(f"  {len(summary['algorithms'])} algorithms")
print(f"  {len(summary['per_subphase'])} sub-phases")
print(f"  {sum(len(v['results']) for v in summary['pass_fail'].values())} pass/fail evaluations")
print(f"  {sum(len(v) for v in summary['rankings'].values())} rankings")

# print compact overview
print(f"\n{'='*90}")
print(f"  COMPACT OVERVIEW")
print(f"{'='*90}")
for sp in SUB_PHASES:
    print(f"\n  {sp.upper()}")
    print(f"  {'Algorithm':<14} {'Collision':>10} {'Plan Time':>12} {'Success':>10} {'Result':>8}")
    print(f"  {'-'*60}")
    pf = summary["pass_fail"][sp]["results"]
    for algo in test_algos:
        if algo not in pf:
            continue
        r = pf[algo]
        print(f"  {algo:<14} {r.get('collision_rate_pct', 'N/A'):>8}% "
              f"{r.get('planning_time_s', 0):>10.3f}s "
              f"{r.get('success_pct', 0):>7.1f}% "
              f"{r['overall']:>8}")
print()
