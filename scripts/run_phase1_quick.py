import os, json, time
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
N_CASES = 5  # change to None for full 50

# A* is pre-computed in test cases — skip it during benchmark
ALL_PLANNERS = [
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
for f in [csv_path, json_path]:
    if os.path.exists(f):
        os.remove(f)

logger = TrialLogger(csv_path)
all_rows = []
tid = 1
t_global = time.perf_counter()

for sp in SUB_PHASES:
    w = PandaWorld(gui=False, acm_samples=200)
    load_obstacles(w, f"config/{sp}_obstacles.json")
    cases = load_test_cases(f"config/{sp}_test_cases.json")
    if N_CASES:
        cases = cases[:N_CASES]

    print(f"\n{'='*80}")
    print(f"  {sp.upper()} | {len(cases)} cases | {len(ALL_PLANNERS)} algorithms")
    print(f"  d_optimal pre-computed by A* (stored in test_cases.json)")
    print(f"{'='*80}")

    # inject A* baseline rows from pre-computed data
    for ci, case in enumerate(cases):
        d_opt = case.get("d_optimal")
        a_row = dict(
            trial_id=tid, sub_phase=sp, algorithm="a_star",
            test_case_id=case["test_case_id"], trial_number=ci+1, random_seed=None,
            planning_success=True,
            path_planning_time=case.get("astar_planning_time", 0),
            path_execution_time=None, total_time=None,
            path_smoothness_msj=None, path_distance_raw=None,
            path_distance_optimal=d_opt, path_optimality_ratio=1.0,
            collision_flag=False, num_contact_points=0,
            algorithm_specific_metadata={"method": case.get("astar_method", "pre-computed"),
                                         "d_optimal": d_opt})
        logger.log(a_row)
        all_rows.append(a_row)
        tid += 1

    n_astar = len(cases)
    print(f"  a_star         success={n_astar:>2}/{len(cases)}  collision=  0.0%  (pre-computed)")

    for algo_name, planner in ALL_PLANNERS:
        t_algo = time.perf_counter()
        successes, collisions = 0, 0

        for ci, case in enumerate(cases):
            d_opt = case.get("d_optimal")
            row = run_trial(w, planner, case, d_opt, tid, sp, ci+1)
            logger.log(row)
            all_rows.append(row)
            if row["planning_success"]:
                successes += 1
            if row.get("collision_flag"):
                collisions += 1
            tid += 1

        elapsed = time.perf_counter() - t_algo
        col_rate = collisions / max(successes, 1) * 100
        print(f"  {algo_name:<14} success={successes:>2}/{len(cases)}  collision={col_rate:>5.1f}%  time={elapsed:>6.1f}s")

    w.disconnect()

logger.close()
with open(json_path, "w") as f:
    json.dump(all_rows, f, indent=2, default=str)

total_time = time.perf_counter() - t_global

# ---- summary ----
df = pd.DataFrame(all_rows)
algos = ["a_star", "min_jerk"] + [n for n, _ in ALL_PLANNERS]
mj_msj = df[df["algorithm"] == "min_jerk"]["path_smoothness_msj"].dropna().mean()

print(f"\n{'='*80}")
print(f"  PHASE 1 COMPLETE — {tid-1} trials in {total_time/60:.1f} min")
print(f"{'='*80}")

for sp in SUB_PHASES:
    sp_df = df[df["sub_phase"] == sp]
    mj_col = sp_df[sp_df["algorithm"] == "min_jerk"]["collision_flag"].sum()
    mj_col_rate = mj_col / len(sp_df[sp_df["algorithm"] == "min_jerk"]) * 100
    as_pt = sp_df[sp_df["algorithm"] == "a_star"]["path_planning_time"].mean()

    print(f"\n  {sp.upper()} (floor coll={mj_col_rate:.0f}%, ceiling plan={as_pt:.2f}s)")
    print(f"  {'Algorithm':<14} {'Success':>8} {'Collision':>10} {'Plan Time':>12} {'Result':>8}")
    print(f"  {'-'*60}")

    for algo in algos:
        sub = sp_df[sp_df["algorithm"] == algo]
        if len(sub) == 0:
            continue
        sub_ok = sub[sub["planning_success"] == True]
        success = len(sub_ok)
        col = sub_ok["collision_flag"].sum() / max(len(sub_ok), 1) * 100 if len(sub_ok) > 0 else 0
        pt = sub["path_planning_time"].mean()
        success_pct = success / len(sub) * 100

        if algo in ("a_star", "min_jerk"):
            result = "BASE"
        else:
            col_pass = col < mj_col_rate or mj_col_rate == 0
            pt_pass = pt < as_pt
            result = "PASS" if col_pass and pt_pass and success_pct >= 80 else "FAIL"

        print(f"  {algo:<14} {success:>3}/{len(sub)} {col:>8.1f}% {pt:>10.3f}s {result:>8}")

print(f"\n  saved: {csv_path} | {json_path}")
