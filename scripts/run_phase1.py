import os, json, time, sys
os.environ["NANOBIND_LEAK_WARNINGS"] = "0"

import numpy as np
import pandas as pd
from arm_benchmark.core.world import PandaWorld
from arm_benchmark.scenes.obstacles import load_obstacles
from arm_benchmark.scenes.testcases import load_test_cases
from arm_benchmark.planners.min_jerk import MinJerk
from arm_benchmark.planners.astar import AStar
from arm_benchmark.planners.ompl_planners import OMPLPlanner
from arm_benchmark.planners.apf import APF
from arm_benchmark.planners.chomp import CHOMP
from arm_benchmark.planners.stomp import STOMP
from arm_benchmark.runner import run_trial
from arm_benchmark.logging.logger import TrialLogger

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SUB_PHASES = ["sparse", "moderate", "dense"]

ALL_PLANNERS = [
    ("a_star", AStar()),
    ("min_jerk", MinJerk()),
    ("RRT", OMPLPlanner("RRT", time_budget=5.0)),
    ("RRT-Connect", OMPLPlanner("RRT-Connect", time_budget=5.0)),
    ("RRT*", OMPLPlanner("RRT*", time_budget=5.0)),
    ("PRM", OMPLPlanner("PRM", time_budget=5.0)),
    ("APF", APF()),
    ("CHOMP", CHOMP()),
    ("STOMP", STOMP()),
]

csv_path = os.path.join(RESULTS_DIR, "phase1_results.csv")
json_path = os.path.join(RESULTS_DIR, "phase1_results.json")

# clear old results
for f in [csv_path, json_path]:
    if os.path.exists(f):
        os.remove(f)

logger = TrialLogger(csv_path)
all_rows = []
tid = 1
t_start = time.perf_counter()

for sp in SUB_PHASES:
    obs_file = f"config/{sp}_obstacles.json"
    cases_file = f"config/{sp}_test_cases.json"

    w = PandaWorld(gui=False, acm_samples=200)
    load_obstacles(w, obs_file)
    cases = load_test_cases(cases_file)

    d_optimals = {}

    print(f"\n{'='*80}")
    print(f"  SUB-PHASE: {sp.upper()} | {len(cases)} cases | {len(ALL_PLANNERS)} algorithms")
    print(f"{'='*80}")

    for algo_name, planner in ALL_PLANNERS:
        t_algo = time.perf_counter()
        successes, collisions, total = 0, 0, 0

        for ci, case in enumerate(cases):
            d_opt = d_optimals.get(case["test_case_id"])
            row = run_trial(w, planner, case, d_opt, tid, sp, ci + 1)
            logger.log(row)
            all_rows.append(row)

            if algo_name == "a_star" and row.get("path_distance_optimal"):
                d_optimals[case["test_case_id"]] = row["path_distance_optimal"]

            total += 1
            if row["planning_success"]:
                successes += 1
            if row.get("collision_flag"):
                collisions += 1
            tid += 1

        elapsed = time.perf_counter() - t_algo
        col_rate = collisions / max(successes, 1) * 100
        print(f"  {algo_name:<14} success={successes:>2}/{total}  collision={col_rate:>5.1f}%  time={elapsed:>6.1f}s")

    w.disconnect()

logger.close()

with open(json_path, "w") as f:
    json.dump(all_rows, f, indent=2, default=str)

total_time = time.perf_counter() - t_start
print(f"\n{'='*80}")
print(f"  PHASE 1 COMPLETE — {tid-1} trials in {total_time/60:.1f} min")
print(f"  CSV: {csv_path} | JSON: {json_path}")
print(f"{'='*80}")

# ---- summary ----
df = pd.DataFrame(all_rows)
mj_msj = df[df["algorithm"] == "min_jerk"]["path_smoothness_msj"].dropna().mean()

print(f"\n{'='*130}")
print(f"  PHASE 1 FULL SUMMARY")
print(f"{'='*130}")

for sp in SUB_PHASES:
    sp_df = df[df["sub_phase"] == sp]
    print(f"\n  --- {sp.upper()} ---")
    print(f"  {'Algorithm':<14} {'MSJ':>12} {'MSJ ratio':>10} {'Plan Time':>12} "
          f"{'Exec Time':>12} {'POR':>8} {'Collision':>10} {'Success':>10}")
    print("  " + "-" * 100)

    for algo_name, _ in ALL_PLANNERS:
        sub = sp_df[sp_df["algorithm"] == algo_name]
        if len(sub) == 0: continue
        ok = sub[sub["planning_success"] == True]
        msj = ok["path_smoothness_msj"].dropna()
        pt = sub["path_planning_time"].dropna()
        et = ok["path_execution_time"].dropna()
        por = ok["path_optimality_ratio"].dropna()
        col = ok["collision_flag"].sum() / max(len(ok), 1) * 100
        ratio = msj.mean() / mj_msj if mj_msj > 0 and len(msj) > 0 else 0
        print(f"  {algo_name:<14} "
              f"{msj.mean() if len(msj) else 0:>10.1f} "
              f"{ratio:>8.1f}x "
              f"{pt.mean():>10.4f} "
              f"{et.mean() if len(et) else 0:>10.3f} "
              f"{por.mean() if len(por) else 0:>6.4f} "
              f"{col:>8.1f}% "
              f"{int(ok.shape[0]):>5}/{len(sub)}")

# ---- pass/fail per sub-phase ----
print(f"\n{'='*130}")
print(f"  PASS/FAIL PER SUB-PHASE")
print(f"{'='*130}")

for sp in SUB_PHASES:
    sp_df = df[df["sub_phase"] == sp]
    mj_col = sp_df[sp_df["algorithm"] == "min_jerk"]["collision_flag"].sum()
    mj_col = mj_col / len(sp_df[sp_df["algorithm"] == "min_jerk"]) * 100
    as_pt = sp_df[sp_df["algorithm"] == "a_star"]["path_planning_time"].mean()

    print(f"\n  {sp.upper()} (floor collision={mj_col:.1f}%, ceiling plan={as_pt:.2f}s)")
    print(f"  {'Algorithm':<14} {'Coll':>8} {'vs MJ':>8} {'Plan':>8} {'vs A*':>8} {'Success':>8}  {'RESULT':>8}")
    print("  " + "-" * 70)

    for algo_name, _ in ALL_PLANNERS:
        if algo_name in ("a_star", "min_jerk"): continue
        sub = sp_df[sp_df["algorithm"] == algo_name]
        ok = sub[sub["planning_success"] == True]
        if len(ok) == 0:
            print(f"  {algo_name:<14} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'0%':>8}  {'FAIL':>8}")
            continue
        col = ok["collision_flag"].sum() / len(ok) * 100
        pt = sub["path_planning_time"].mean()
        success_pct = len(ok) / len(sub) * 100
        col_pass = col < mj_col or mj_col == 0
        pt_pass = pt < as_pt
        overall = col_pass and pt_pass and success_pct >= 80
        print(f"  {algo_name:<14} {col:>6.1f}% {'PASS' if col_pass else 'FAIL':>8} "
              f"{pt:>6.2f}s {'PASS' if pt_pass else 'FAIL':>8} "
              f"{success_pct:>5.0f}%  {'PASS' if overall else 'FAIL':>8}")

print()
