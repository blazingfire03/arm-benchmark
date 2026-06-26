import json, os
import numpy as np
import pandas as pd

# load data
df = pd.read_csv("results/phase1_results.csv")
with open("results/phase1_summary.json") as f:
    summary = json.load(f)

ok = df[df["planning_success"] == True].copy()

SUB_PHASES = ["sparse", "moderate", "dense"]
ALGORITHMS = ["a_star", "min_jerk", "RRT", "RRT-Connect", "RRT*", "PRM", "APF", "CHOMP", "STOMP"]
OBS_COUNTS = {"sparse": 4, "moderate": 12, "dense": 16}

print("=" * 120)
print("  PHASE 1 COMPLETE ANALYSIS — PRESENTATION SUMMARY")
print("=" * 120)

# ============================================================
# TABLE 1: Main Results Table (for PPT slide)
# ============================================================
print("\n\n  TABLE 1: MAIN RESULTS (mean values, 5 trials per cell)")
print("=" * 120)

for sp in SUB_PHASES:
    sp_df = df[df["sub_phase"] == sp]
    sp_ok = ok[ok["sub_phase"] == sp]
    print(f"\n  {sp.upper()} SUB-PHASE ({OBS_COUNTS[sp]} obstacles)")
    print(f"  {'Algorithm':<14} {'Success':>8} {'Collision':>10} {'Plan Time':>12} {'Exec Time':>12} {'MSJ':>14} {'POR':>8}")
    print(f"  {'-'*82}")

    for algo in ALGORITHMS:
        sub = sp_df[sp_df["algorithm"] == algo]
        sub_ok = sp_ok[sp_ok["algorithm"] == algo]
        if len(sub) == 0:
            continue
        n = len(sub)
        success = int(sub["planning_success"].sum())
        success_pct = success / n * 100

        if len(sub_ok) > 0:
            col = sub_ok["collision_flag"].sum() / len(sub_ok) * 100
            msj = sub_ok["path_smoothness_msj"].dropna()
            pt = sub["path_planning_time"].dropna()
            et = sub_ok["path_execution_time"].dropna()
            por = sub_ok["path_optimality_ratio"].dropna()

            msj_str = f"{msj.mean():.1f}" if len(msj) > 0 else "—"
            pt_str = f"{pt.mean():.4f}" if len(pt) > 0 else "—"
            et_str = f"{et.mean():.3f}" if len(et) > 0 else "—"
            por_str = f"{por.mean():.4f}" if len(por) > 0 and not np.isnan(por.mean()) else "—"
        else:
            col = 0
            msj_str = pt_str = et_str = por_str = "—"
            pt = sub["path_planning_time"].dropna()
            pt_str = f"{pt.mean():.4f}" if len(pt) > 0 else "—"

        print(f"  {algo:<14} {success:>3}/{n} ({success_pct:>3.0f}%) {col:>7.1f}% {pt_str:>12} {et_str:>12} {msj_str:>14} {por_str:>8}")

# ============================================================
# TABLE 2: Pass/Fail Summary (for PPT slide)
# ============================================================
print("\n\n  TABLE 2: PASS/FAIL EVALUATION")
print("=" * 120)
print(f"  {'Algorithm':<14} {'Sparse':>10} {'Moderate':>10} {'Dense':>10} {'Passed':>10} {'Recommendation':>25}")
print(f"  {'-'*82}")

for algo in ["RRT", "RRT-Connect", "RRT*", "PRM", "APF", "CHOMP", "STOMP"]:
    results = []
    for sp in SUB_PHASES:
        sp_df = df[df["sub_phase"] == sp]
        sp_ok = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == algo)]
        mj_sub = sp_df[sp_df["algorithm"] == "min_jerk"]; mj_col = mj_sub["collision_flag"].sum() / max(len(mj_sub), 1) * 100
        as_pt = sp_df[sp_df["algorithm"] == "a_star"]["path_planning_time"].mean()

        sub = sp_df[sp_df["algorithm"] == algo]
        if len(sub) == 0 or len(sp_ok) == 0:
            results.append("FAIL")
            continue

        col = sp_ok["collision_flag"].sum() / len(sp_ok) * 100
        pt = sub["path_planning_time"].mean()
        success_pct = sub["planning_success"].sum() / len(sub) * 100

        col_pass = col < mj_col or mj_col == 0
        pt_pass = pt < as_pt
        passed = col_pass and pt_pass and success_pct >= 80
        results.append("PASS" if passed else "FAIL")

    n_pass = results.count("PASS")
    rec = "ADVANCE" if n_pass >= 2 else "EXCLUDE"
    print(f"  {algo:<14} {results[0]:>10} {results[1]:>10} {results[2]:>10} {n_pass:>5}/3     {rec:>20}")

# ============================================================
# TABLE 3: Key Findings
# ============================================================
print("\n\n  TABLE 3: KEY FINDINGS PER ALGORITHM")
print("=" * 120)

findings = {
    "RRT": {
        "type": "Sampling-based",
        "strengths": "Fast planning (~0.1s), 0% collision in sparse/moderate, 100% success sparse",
        "weaknesses": "High MSJ (rough paths), 40% success in dense",
        "verdict": "ADVANCE — strong all-rounder"
    },
    "RRT-Connect": {
        "type": "Sampling-based (bidirectional)",
        "strengths": "Fastest planner (~0.04s), 0% collision sparse, 100% success sparse/moderate",
        "weaknesses": "40-50% collision in dense, some collisions in moderate",
        "verdict": "ADVANCE — fastest, best for real-time"
    },
    "RRT*": {
        "type": "Sampling-based (optimal)",
        "strengths": "Asymptotically optimal paths, 0% collision sparse, lower MSJ than RRT",
        "weaknesses": "Fixed 5s planning time (uses full budget), 60% success in dense",
        "verdict": "ADVANCE — best path quality among sampling planners"
    },
    "PRM": {
        "type": "Roadmap-based",
        "strengths": "Fast (~0.04s), 0% collision sparse/moderate, 100% success sparse/moderate",
        "weaknesses": "50% collision in dense, 40% success in dense",
        "verdict": "ADVANCE — excellent for repeated queries"
    },
    "APF": {
        "type": "Reactive (gradient)",
        "strengths": "Very fast when it works, simple implementation, 0% collision sparse",
        "weaknesses": "0% success in moderate/dense (local minima), not viable for cluttered spaces",
        "verdict": "EXCLUDE — local minima make it unreliable"
    },
    "CHOMP": {
        "type": "Optimization (gradient-based)",
        "strengths": "0% collision sparse/moderate (with RRT-Connect seed), smooths rough paths",
        "weaknesses": "Slow (4-7s), high MSJ, 60% collision moderate when seed is poor",
        "verdict": "ADVANCE — good smoother, needs collision-free seed"
    },
    "STOMP": {
        "type": "Optimization (stochastic)",
        "strengths": "0% collision sparse (with seed), can escape local optima",
        "weaknesses": "Slowest planner (7-10s), fails sparse speed test, 33% collision dense",
        "verdict": "EXCLUDE — too slow, marginal improvement over CHOMP"
    },
}

for algo, info in findings.items():
    print(f"\n  {algo} ({info['type']})")
    print(f"    Strengths:  {info['strengths']}")
    print(f"    Weaknesses: {info['weaknesses']}")
    print(f"    Verdict:    {info['verdict']}")

# ============================================================
# TABLE 4: Figure Analysis
# ============================================================
print("\n\n  TABLE 4: FIGURE-BY-FIGURE ANALYSIS")
print("=" * 120)

analyses = [
    ("Success Rate (Fig 1)", [
        "A* degrades: 80% sparse → 20% moderate → 0% dense (grid search intractable in clutter)",
        "Min-jerk always 100% (trivial polynomial, no search needed)",
        "RRT/RRT-Connect/PRM maintain 100% through moderate, drop to 40-50% in dense",
        "APF collapses to 0% in moderate/dense (local minima)",
        "CHOMP/STOMP maintain 100% in sparse/moderate (RRT-Connect seed ensures a path)",
    ]),
    ("MSJ Smoothness (Fig 2)", [
        "Min-jerk is the smoothness floor: MSJ ≈ 1 rad²/s⁵ across all sub-phases",
        "A* matches min-jerk in sparse (same straight-line path) but absent in moderate/dense",
        "OMPL planners: MSJ ≈ 10⁶ in moderate/dense — sharp corners from collision avoidance",
        "APF: MSJ ≈ 10⁵ in sparse — gradient-following creates oscillations near obstacles",
        "Key finding: collision avoidance costs 6 orders of magnitude in smoothness",
    ]),
    ("Planning Time (Fig 3)", [
        "Min-jerk: ~0.1ms (near-instant, just polynomial evaluation)",
        "RRT-Connect/PRM: ~40-100ms (fastest obstacle-aware planners)",
        "RRT: ~100ms sparse, slower in clutter",
        "RRT*: fixed ~5s (uses entire budget for path optimization)",
        "CHOMP: 0.5-5s (gradient iterations), STOMP: 7-10s (stochastic rollouts)",
        "A*: 0.03s sparse (direct path), 24s+ moderate (grid search), timeout dense",
    ]),
    ("Execution Time (Fig 4)", [
        "Consistent 4-7s across most algorithms in sparse (similar path lengths after retiming)",
        "Increases with density: moderate 5-8s, dense 5-16s",
        "STOMP has highest execution time in dense (~12-16s) — noisy paths are longer",
        "Min-jerk execution time is stable (straight-line paths have consistent length)",
    ]),
    ("Collision Rate (Fig 5)", [
        "Min-jerk: 20% sparse → 80% moderate → 100% dense (validates calibration targets)",
        "RRT/RRT-Connect/PRM: 0% sparse, 0-20% moderate — effective collision avoidance",
        "Dense: RRT-Connect 50%, PRM 50%, RRT* 33% — even good planners struggle",
        "CHOMP: 0% in sparse+moderate (collision constraint works), 0% in dense cases it solves",
        "APF: 0% collision when it succeeds, but 0% success in moderate/dense",
    ]),
    ("Trade-off Scatter (Fig 6)", [
        "Ideal position: bottom-left (fast planning, low collision)",
        "RRT-Connect and PRM are closest to ideal in sparse and moderate",
        "Min-jerk: fast but high collision — the floor baseline working as designed",
        "RRT*: moderate speed, low collision — good balance",
        "STOMP: slow and still has collisions in dense — worst trade-off",
    ]),
    ("Radar Plots (Fig 7-9)", [
        "Sparse: most algorithms cover similar area, APF loses on smoothness",
        "Moderate: RRT-Connect has largest radar area (best overall balance)",
        "Dense: min-jerk dominates smoothness/speed but fails safety completely",
        "No algorithm dominates all 5 axes — confirms multi-objective nature of the comparison",
    ]),
    ("POR (Fig 10)", [
        "Most algorithms: POR ≈ 1.0 (matching A* optimal path length)",
        "APF: POR ≈ 1.1 in sparse (10% longer paths due to repulsive force detours)",
        "Dense POR empty — A* couldn't solve those cases, so no d_optimal reference",
        "POR is less discriminating than expected — the real differentiation is in collision rate",
    ]),
]

for title, points in analyses:
    print(f"\n  {title}")
    for pt in points:
        print(f"    • {pt}")

# ============================================================
# TABLE 5: Phase 2 Recommendations
# ============================================================
print("\n\n  TABLE 5: PHASE 2 RECOMMENDATIONS")
print("=" * 120)
print(f"  {'Algorithm':<14} {'Sub-phases':>12} {'Decision':>10} {'Rationale':<60}")
print(f"  {'-'*100}")

recs = [
    ("RRT",         "2/3", "ADVANCE",  "Fast, reliable, 0% collision in sparse/moderate"),
    ("RRT-Connect", "2/3", "ADVANCE",  "Fastest planner, best real-time candidate for RAMMP"),
    ("RRT*",        "2/3", "ADVANCE",  "Best path quality, worth testing with dynamic obstacles"),
    ("PRM",         "2/3", "ADVANCE",  "Fast roadmap, 0% collision moderate, good for repeated queries"),
    ("CHOMP",       "2/3", "ADVANCE",  "Smoothes paths effectively with collision-free seed"),
    ("APF",         "1/3", "EXCLUDE",  "Local minima kill it in moderate+ density"),
    ("STOMP",       "1/3", "EXCLUDE",  "Too slow, marginal benefit over CHOMP"),
]

for algo, sp, dec, rationale in recs:
    print(f"  {algo:<14} {sp:>12} {dec:>10}   {rationale:<60}")

# ============================================================
# Save as JSON for PPT generation
# ============================================================
ppt_data = {
    "title": "Phase 1 Results: Evaluation of Collision-Aware Algorithms for the 7-DOF Franka Emika Panda Arm",
    "experiment": {
        "robot": "Franka Emika Panda 7-DOF",
        "simulator": "PyBullet",
        "trials_per_cell": 5,
        "sub_phases": {"sparse": "4 obstacles", "moderate": "12 obstacles", "dense": "16 obstacles"},
        "baselines": {"floor": "min-jerk (obstacle-blind)", "ceiling": "A* (optimal collision-free)"},
    },
    "main_results": {},
    "pass_fail": {},
    "findings": findings,
    "figure_analyses": {title: points for title, points in analyses},
    "recommendations": recs,
    "key_takeaways": [
        "RRT-Connect is the best overall performer: fastest planning, 0% collision in sparse/moderate, 100% success through moderate",
        "PRM matches RRT-Connect on speed and safety, better suited for repeated planning queries",
        "CHOMP effectively smoothes rough planner output when given a collision-free RRT-Connect seed",
        "APF is fundamentally limited by local minima — not viable beyond sparse environments",
        "Dense environments (16 obstacles) are genuinely hard: even the best planners achieve only 40-60% success",
        "Grid-based A* is intractable for 7-DOF dense planning — sampling-based methods are strictly necessary",
        "MSJ differs by 6 orders of magnitude between baselines and obstacle-aware planners — smoothness is the primary cost of collision avoidance",
        "5 of 7 algorithms advance to Phase 2: RRT, RRT-Connect, RRT*, PRM, CHOMP",
    ],
}

# fill main results
for sp in SUB_PHASES:
    sp_data = {}
    for algo in ALGORITHMS:
        sub = df[(df["sub_phase"] == sp) & (df["algorithm"] == algo)]
        sub_ok = ok[(ok["sub_phase"] == sp) & (ok["algorithm"] == algo)]
        if len(sub) == 0:
            continue
        sp_data[algo] = {
            "success_rate": round(sub["planning_success"].sum() / len(sub) * 100, 1),
            "collision_rate": round(sub_ok["collision_flag"].sum() / max(len(sub_ok), 1) * 100, 1) if len(sub_ok) > 0 else None,
            "mean_planning_time": round(float(sub["path_planning_time"].mean()), 4),
            "mean_execution_time": round(float(sub_ok["path_execution_time"].dropna().mean()), 3) if len(sub_ok["path_execution_time"].dropna()) > 0 else None,
            "mean_msj": round(float(sub_ok["path_smoothness_msj"].dropna().mean()), 2) if len(sub_ok["path_smoothness_msj"].dropna()) > 0 else None,
            "mean_por": round(float(sub_ok["path_optimality_ratio"].dropna().mean()), 4) if len(sub_ok["path_optimality_ratio"].dropna()) > 0 and not np.isnan(sub_ok["path_optimality_ratio"].dropna().mean()) else None,
        }
    ppt_data["main_results"][sp] = sp_data

os.makedirs("results", exist_ok=True)

class NE(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return super().default(o)

with open("results/ppt_summary.json", "w") as f:
    json.dump(ppt_data, f, indent=2, cls=NE)

print(f"\n\n{'='*120}")
print(f"  SAVED: results/ppt_summary.json")
print(f"  Use this JSON to populate your PPT slides")
print(f"{'='*120}")
