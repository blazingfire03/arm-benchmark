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

OBSTACLES = "config/sparse_obstacles.json"
TESTCASES = "config/sparse_test_cases.json"
RESULTS_CSV = "results/phase1_sparse.csv"
RESULTS_JSON = "results/phase1_sparse.json"
SUB_PHASE = "sparse"
N_CASES = 10

os.makedirs("results", exist_ok=True)
for f in [RESULTS_CSV, RESULTS_JSON]:
    if os.path.exists(f):
        os.remove(f)

w = PandaWorld(gui=False)
load_obstacles(w, OBSTACLES)
cases = load_test_cases(TESTCASES)[:N_CASES]

logger = TrialLogger(RESULTS_CSV)
all_rows = []
tid = 1

ALL_PLANNERS = [
    ("a_star", "ceiling", AStar()),
    ("min_jerk", "floor", MinJerk()),
    ("RRT", "test", OMPLPlanner("RRT", time_budget=5.0)),
    ("RRT-Connect", "test", OMPLPlanner("RRT-Connect", time_budget=5.0)),
    ("RRT*", "test", OMPLPlanner("RRT*", time_budget=5.0)),
    ("PRM", "test", OMPLPlanner("PRM", time_budget=5.0)),
    ("APF", "test", APF()),
    ("CHOMP", "test", CHOMP()),
    ("STOMP", "test", STOMP()),
]

def fmt(val, width=10, decimals=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"{'N/A':>{width}}"
    return f"{val:>{width}.{decimals}f}"

print(f"PHASE 1 BENCHMARK — {SUB_PHASE.upper()} | {len(cases)} cases | {len(ALL_PLANNERS)} algorithms\n")

d_optimals = {}
for algo_name, role, planner in ALL_PLANNERS:
    print(f"{algo_name.upper()} ({role})")
    print(f"{'case':>5} {'success':>8} {'plan_t':>10} {'exec_t':>10} {'MSJ':>12} {'POR':>8} {'coll':>6}")
    print("-" * 68)
    for ci, case in enumerate(cases):
        d_opt = d_optimals.get(case["test_case_id"])
        row = run_trial(w, planner, case, d_opt, tid, SUB_PHASE, ci + 1)
        logger.log(row); all_rows.append(row)
        if algo_name == "a_star" and row["path_distance_optimal"]:
            d_optimals[case["test_case_id"]] = row["path_distance_optimal"]

        ok = row["planning_success"]
        pt = row["path_planning_time"]
        et = row.get("path_execution_time")
        msj = row.get("path_smoothness_msj")
        por = row.get("path_optimality_ratio")
        coll = row.get("collision_flag")

        print(f"{ci:>5} {'OK' if ok else 'FAIL':>8} "
              f"{fmt(pt, 10, 4)} "
              f"{fmt(et, 10, 3)} "
              f"{fmt(msj, 12, 2)} "
              f"{fmt(por, 8, 4)} "
              f"{'YES' if coll else 'no' if coll is not None else 'N/A':>6}")
        tid += 1
    print()

logger.close()
w.disconnect()

with open(RESULTS_JSON, "w") as f:
    json.dump(all_rows, f, indent=2, default=str)

# ---- summary ----
df = pd.DataFrame(all_rows)
algos = [name for name, _, _ in ALL_PLANNERS]
mj_msj = df[df["algorithm"] == "min_jerk"]["path_smoothness_msj"].dropna().mean()

print("=" * 125)
print(f"  PHASE 1 SUMMARY — {SUB_PHASE.upper()} ({len(cases)} trials per algorithm)")
print("=" * 125)
print(f"  {'Algorithm':<14} {'Role':<8} {'MSJ':>14} {'MSJ ratio':>10} {'Plan Time':>14} "
      f"{'Exec Time':>14} {'POR':>10} {'Collision':>10} {'Success':>10}")
print("-" * 125)

for algo in algos:
    sub = df[df["algorithm"] == algo]
    if len(sub) == 0: continue
    role = "ceiling" if algo == "a_star" else "floor" if algo == "min_jerk" else "test"
    ok = sub[sub["planning_success"] == True]
    msj = ok["path_smoothness_msj"].dropna()
    pt = sub["path_planning_time"].dropna()
    et = ok["path_execution_time"].dropna()
    por = ok["path_optimality_ratio"].dropna()
    col = ok["collision_flag"].sum() / max(len(ok), 1) * 100
    ratio = msj.mean() / mj_msj if mj_msj > 0 and len(msj) > 0 else 0
    print(f"  {algo:<14} {role:<8} "
          f"{msj.mean() if len(msj) else 0:>7.1f}±{msj.std() if len(msj) else 0:>5.1f} "
          f"{ratio:>8.1f}x "
          f"{pt.mean():>7.4f}±{pt.std():>5.4f} "
          f"{et.mean() if len(et) else 0:>7.3f}±{et.std() if len(et) else 0:>5.3f} "
          f"{por.mean() if len(por) else 0:>6.4f} "
          f"{col:>7.1f}% "
          f"{int(ok.shape[0]):>5}/{len(sub)}")
print("-" * 125)

# ---- PASS/FAIL ----
mj_col = df[df["algorithm"] == "min_jerk"]["collision_flag"].sum() / len(cases) * 100
as_pt = df[df["algorithm"] == "a_star"]["path_planning_time"].mean()

print(f"\n  PHASE 1 PASS/FAIL")
print(f"  {'Algorithm':<14} {'Coll':>8} {'vs MJ':>8} {'Plan':>8} {'vs A*':>8} {'Success':>8} {'MSJ ratio':>10}  {'RESULT':>8}")
print("  " + "-" * 80)

for algo in ["RRT", "RRT-Connect", "RRT*", "PRM", "APF", "CHOMP", "STOMP"]:
    sub = df[df["algorithm"] == algo]
    if len(sub) == 0: continue
    ok = sub[sub["planning_success"] == True]
    if len(ok) == 0:
        print(f"  {algo:<14} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'0%':>8} {'N/A':>10}  {'FAIL':>8}")
        continue
    col = ok["collision_flag"].sum() / len(ok) * 100
    pt = sub["path_planning_time"].mean()
    msj = ok["path_smoothness_msj"].dropna().mean()
    ratio = msj / mj_msj if mj_msj > 0 else 0
    success_pct = len(ok) / len(sub) * 100
    col_pass = col < mj_col or mj_col == 0
    pt_pass = pt < as_pt
    overall = col_pass and pt_pass and success_pct >= 80
    print(f"  {algo:<14} {col:>6.1f}% {'PASS' if col_pass else 'FAIL':>8} "
          f"{pt:>6.2f}s {'PASS' if pt_pass else 'FAIL':>8} "
          f"{success_pct:>5.0f}% {ratio:>8.1f}x  {'PASS' if overall else 'FAIL':>8}")

print(f"""
  Baselines:
    min_jerk (floor):   collision={mj_col:.1f}%  MSJ={mj_msj:.2f} (1.0x)  plan={df[df['algorithm']=='min_jerk']['path_planning_time'].mean()*1e6:.0f} us
    A* (ceiling):       collision=0.0%   POR=1.0  plan={as_pt:.4f} s
""")
print(f"saved — CSV: {RESULTS_CSV} | JSON: {RESULTS_JSON}")
