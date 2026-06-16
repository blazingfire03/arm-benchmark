import os, json
import numpy as np
import pandas as pd

RESULTS = "results/phase1_results.csv"
OUTPUT = "results/phase1_summary.json"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

df = pd.read_csv(RESULTS)
ok = df[df["planning_success"] == True].copy()

SUB_PHASES = ["sparse", "moderate", "dense"]
ALGORITHMS = ["a_star", "min_jerk", "RRT", "RRT-Connect", "RRT*", "PRM", "APF", "CHOMP", "STOMP"]
TEST_ALGOS = ["RRT", "RRT-Connect", "RRT*", "PRM", "APF", "CHOMP", "STOMP"]
ROLES = {"a_star": "ceiling_baseline", "min_jerk": "floor_baseline",
         "RRT": "sampling_based", "RRT-Connect": "sampling_based",
         "RRT*": "sampling_based", "PRM": "roadmap",
         "APF": "reactive", "CHOMP": "optimization", "STOMP": "optimization"}
PARADIGMS = {"RRT": "sampling", "RRT-Connect": "sampling", "RRT*": "sampling",
             "PRM": "roadmap", "APF": "reactive", "CHOMP": "optimization", "STOMP": "optimization"}

def stats(vals):
    if len(vals) == 0:
        return None
    return {"mean": round(float(vals.mean()), 6), "std": round(float(vals.std()), 6),
            "median": round(float(vals.median()), 6), "min": round(float(vals.min()), 6),
            "max": round(float(vals.max()), 6), "p25": round(float(vals.quantile(0.25)), 6),
            "p75": round(float(vals.quantile(0.75)), 6), "count": int(len(vals))}

def algo_metrics(sub_all, sub_ok):
    n_total = len(sub_all)
    n_success = int(sub_all["planning_success"].sum())
    n_collision = int(sub_ok["collision_flag"].sum()) if len(sub_ok) > 0 else 0
    return {
        "trials": n_total,
        "success_count": n_success,
        "success_rate_pct": round(n_success / max(n_total, 1) * 100, 1),
        "collision_count": n_collision,
        "collision_rate_pct": round(n_collision / max(n_success, 1) * 100, 1),
        "metrics": {
            "path_smoothness_msj": stats(sub_ok["path_smoothness_msj"].dropna()),
            "path_planning_time": stats(sub_all["path_planning_time"].dropna()),
            "path_execution_time": stats(sub_ok["path_execution_time"].dropna()),
            "path_optimality_ratio": stats(sub_ok["path_optimality_ratio"].dropna()),
        }
    }

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
        "total_trials": len(df),
        "metrics_explanation": {
            "MSJ": "Mean Squared Jerk (rad²/s⁵) — lower is smoother. Min-jerk is the theoretical minimum.",
            "planning_time": "Wall-clock time to compute path (s) — lower is faster. Min-jerk is near-instant.",
            "execution_time": "Time to traverse planned path in simulation (s) — lower means shorter/faster path.",
            "POR": "Path Optimality Ratio = d_path / d_optimal. 1.0 = matched A* optimal. Higher = more detour.",
            "collision_rate": "% of trials where arm hit an obstacle. 0% ideal. A* is 0% by construction.",
            "success_rate": "% of trials where planner found a path. 100% ideal.",
        },
        "baseline_framing": {
            "description": "All 7 algorithms are evaluated between two baselines that bound the performance space.",
            "floor_baseline": "min_jerk — obstacle-blind, mathematically smoothest, near-instant planning, HIGH collision rate.",
            "ceiling_baseline": "A* — optimal collision-free path, provides d_optimal for POR, SLOW planning, 0% collision.",
            "pass_criteria": "collision < min_jerk AND planning_time < A* AND success >= 80%",
        },
    },
}

# ============================================================
# BASELINES
# ============================================================
baselines = {}
for algo in ["a_star", "min_jerk"]:
    entry = {"algorithm": algo, "role": ROLES[algo], "per_subphase": {}}
    for sp in SUB_PHASES:
        sub_all = df[(df["sub_phase"] == sp) & (df["algorithm"] == algo)]
        sub_ok = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == algo)]
        entry["per_subphase"][sp] = algo_metrics(sub_all, sub_ok)
    # overall
    sub_all = df[df["algorithm"] == algo]
    sub_ok = ok[ok["algorithm"] == algo]
    entry["overall"] = algo_metrics(sub_all, sub_ok)
    baselines[algo] = entry
summary["baselines"] = baselines

# ============================================================
# ALGORITHMS WITH BASELINE COMPARISONS
# ============================================================
algorithms = {}
for algo in TEST_ALGOS:
    entry = {
        "algorithm": algo,
        "role": ROLES[algo],
        "paradigm": PARADIGMS[algo],
        "per_subphase": {},
    }

    for sp in SUB_PHASES:
        sub_all = df[(df["sub_phase"] == sp) & (df["algorithm"] == algo)]
        sub_ok = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == algo)]
        mj_all = df[(df["sub_phase"] == sp) & (df["algorithm"] == "min_jerk")]
        mj_ok = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == "min_jerk")]
        as_all = df[(df["sub_phase"] == sp) & (df["algorithm"] == "a_star")]
        as_ok = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == "a_star")]

        result = algo_metrics(sub_all, sub_ok)

        # --- comparison vs min-jerk (floor) ---
        mj_col = mj_ok["collision_flag"].sum() / max(len(mj_ok), 1) * 100
        algo_col = sub_ok["collision_flag"].sum() / max(len(sub_ok), 1) * 100 if len(sub_ok) > 0 else None
        mj_msj = mj_ok["path_smoothness_msj"].dropna().mean() if len(mj_ok) > 0 else None
        algo_msj = sub_ok["path_smoothness_msj"].dropna().mean() if len(sub_ok) > 0 else None
        mj_pt = mj_all["path_planning_time"].mean()
        algo_pt = sub_all["path_planning_time"].mean() if len(sub_all) > 0 else None
        mj_et = mj_ok["path_execution_time"].dropna().mean() if len(mj_ok) > 0 else None
        algo_et = sub_ok["path_execution_time"].dropna().mean() if len(sub_ok) > 0 else None

        vs_floor = {
            "collision_rate": {
                "algorithm": round(float(algo_col), 1) if algo_col is not None else None,
                "min_jerk": round(float(mj_col), 1),
                "improvement_pct": round(float(mj_col - algo_col), 1) if algo_col is not None else None,
                "better_than_floor": bool(algo_col < mj_col) if algo_col is not None else None,
            },
            "MSJ": {
                "algorithm": round(float(algo_msj), 2) if algo_msj is not None and not np.isnan(algo_msj) else None,
                "min_jerk": round(float(mj_msj), 2) if mj_msj is not None and not np.isnan(mj_msj) else None,
                "ratio": round(float(algo_msj / mj_msj), 1) if algo_msj and mj_msj and mj_msj > 0 else None,
                "note": "higher ratio = rougher path (expected for collision-aware planners)",
            },
            "planning_time": {
                "algorithm_s": round(float(algo_pt), 4) if algo_pt is not None else None,
                "min_jerk_s": round(float(mj_pt), 6),
                "slowdown_factor": round(float(algo_pt / mj_pt), 1) if algo_pt and mj_pt > 0 else None,
                "note": "algorithms are expected to be slower than min-jerk",
            },
        }

        # --- comparison vs A* (ceiling) ---
        as_col = as_ok["collision_flag"].sum() / max(len(as_ok), 1) * 100 if len(as_ok) > 0 else 0
        as_pt = as_all["path_planning_time"].mean() if len(as_all) > 0 else None
        as_msj = as_ok["path_smoothness_msj"].dropna().mean() if len(as_ok) > 0 else None
        algo_por = sub_ok["path_optimality_ratio"].dropna().mean() if len(sub_ok) > 0 else None

        vs_ceiling = {
            "collision_rate": {
                "algorithm": round(float(algo_col), 1) if algo_col is not None else None,
                "a_star": round(float(as_col), 1),
                "gap_from_optimal": round(float(algo_col - as_col), 1) if algo_col is not None else None,
                "note": "A* is 0% by construction; closer to 0% = better",
            },
            "planning_time": {
                "algorithm_s": round(float(algo_pt), 4) if algo_pt is not None else None,
                "a_star_s": round(float(as_pt), 4) if as_pt is not None else None,
                "speedup_vs_astar": round(float(as_pt / algo_pt), 1) if algo_pt and as_pt and algo_pt > 0 else None,
                "faster_than_ceiling": bool(algo_pt < as_pt) if algo_pt and as_pt else None,
            },
            "POR": {
                "algorithm": round(float(algo_por), 4) if algo_por is not None and not np.isnan(algo_por) else None,
                "a_star": 1.0,
                "optimality_gap_pct": round(float((algo_por - 1.0) * 100), 1) if algo_por is not None and not np.isnan(algo_por) else None,
                "note": "% extra distance vs A* optimal path",
            },
        }

        # --- pass/fail ---
        col_pass = bool(algo_col < mj_col) if algo_col is not None else False
        pt_pass = bool(algo_pt < as_pt) if algo_pt is not None and as_pt is not None else False
        success_pct = result["success_rate_pct"]
        overall_pass = col_pass and pt_pass and success_pct >= 80

        pass_fail = {
            "collision_vs_floor": {"pass": col_pass, "value": f"{algo_col:.1f}% vs {mj_col:.1f}%" if algo_col is not None else "N/A"},
            "speed_vs_ceiling": {"pass": pt_pass, "value": f"{algo_pt:.2f}s vs {as_pt:.2f}s" if algo_pt and as_pt else "N/A"},
            "success_rate": {"pass": bool(success_pct >= 80), "value": f"{success_pct:.1f}%"},
            "overall": "PASS" if overall_pass else "FAIL",
            "fail_reasons": (
                [] if overall_pass else
                ([f"collision {algo_col:.1f}% >= floor {mj_col:.1f}%"] if not col_pass else []) +
                ([f"plan time {algo_pt:.2f}s >= ceiling {as_pt:.2f}s"] if not pt_pass else []) +
                ([f"success {success_pct:.0f}% < 80%"] if success_pct < 80 else [])
            ),
        }

        result["vs_floor_baseline"] = vs_floor
        result["vs_ceiling_baseline"] = vs_ceiling
        result["pass_fail"] = pass_fail
        entry["per_subphase"][sp] = result

    # overall pass: passes if passes in majority of sub-phases
    passes = sum(1 for sp in SUB_PHASES if entry["per_subphase"].get(sp, {}).get("pass_fail", {}).get("overall") == "PASS")
    entry["overall_verdict"] = {
        "sub_phases_passed": passes,
        "sub_phases_total": len(SUB_PHASES),
        "recommendation": "ADVANCE TO PHASE 2" if passes >= 2 else "EXCLUDE — does not pass enough sub-phases",
    }

    # overall metrics
    sub_all = df[df["algorithm"] == algo]
    sub_ok = ok[ok["algorithm"] == algo]
    entry["overall_metrics"] = algo_metrics(sub_all, sub_ok)

    algorithms[algo] = entry

summary["algorithms"] = algorithms

# ============================================================
# RANKINGS
# ============================================================
rankings = {}
for sp in SUB_PHASES:
    sp_ok = ok[ok["sub_phase"] == sp]
    sp_rankings = {}

    for metric, col, ascending in [
        ("lowest_collision_rate", "collision_flag", True),
        ("fastest_planning", "path_planning_time", True),
        ("smoothest_path", "path_smoothness_msj", True),
        ("shortest_execution", "path_execution_time", True),
    ]:
        scores = []
        for algo in TEST_ALGOS:
            sub = sp_ok[sp_ok["algorithm"] == algo]
            if len(sub) == 0:
                continue
            if col == "collision_flag":
                val = sub[col].sum() / len(sub) * 100
            else:
                vals = sub[col].dropna()
                val = float(vals.mean()) if len(vals) > 0 else float('inf')
            scores.append({"algorithm": algo, "value": round(val, 4)})

        scores.sort(key=lambda x: x["value"])
        sp_rankings[metric] = [{"rank": i+1, **s} for i, s in enumerate(scores)]

    rankings[sp] = sp_rankings

summary["rankings"] = rankings

# ============================================================
# PHASE 2 RECOMMENDATIONS
# ============================================================
recommendations = []
for algo in TEST_ALGOS:
    entry = algorithms[algo]
    verdict = entry["overall_verdict"]
    recommendations.append({
        "algorithm": algo,
        "paradigm": PARADIGMS[algo],
        "sub_phases_passed": verdict["sub_phases_passed"],
        "recommendation": verdict["recommendation"],
        "strengths": [],
        "weaknesses": [],
    })

# sort by sub_phases_passed descending
recommendations.sort(key=lambda x: x["sub_phases_passed"], reverse=True)
summary["phase2_recommendations"] = recommendations

with open(OUTPUT, "w") as f:
    json.dump(summary, f, indent=2, cls=NumpyEncoder)

print(f"saved: {OUTPUT}")
print(f"\nSTRUCTURE:")
for key in summary:
    if isinstance(summary[key], dict):
        print(f"  {key}: {len(summary[key])} entries")
    elif isinstance(summary[key], list):
        print(f"  {key}: {len(summary[key])} items")

print(f"\nPHASE 2 RECOMMENDATIONS:")
for r in recommendations:
    print(f"  {r['algorithm']:<14} passed {r['sub_phases_passed']}/3  →  {r['recommendation']}")

