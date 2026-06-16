import numpy as np
import pandas as pd
from arm_benchmark.core.world import PandaWorld
from arm_benchmark.scenes.obstacles import load_obstacles
from arm_benchmark.scenes.testcases import load_test_cases
from arm_benchmark.planners.min_jerk import MinJerk
from arm_benchmark.planners.astar import AStar
from arm_benchmark.runner import run_trial
from arm_benchmark.logging.logger import TrialLogger

w = PandaWorld(gui=False)
load_obstacles(w, "config/sparse_obstacles.json")
cases = load_test_cases("config/sparse_test_cases.json")[:5]

logger = TrialLogger("results_smoke.csv")
astar = AStar()
tid = 1
for ci, case in enumerate(cases, start=1):
    a_row = run_trial(w, astar, case, None, tid, "sparse", ci); logger.log(a_row); tid += 1
    d_opt = a_row["path_distance_optimal"]
    mj_row = run_trial(w, MinJerk(), case, d_opt, tid, "sparse", ci); logger.log(mj_row); tid += 1
logger.close()
w.disconnect()

df = pd.read_csv("results_smoke.csv")
cols = ["algorithm", "test_case_id", "planning_success", "execution_success",
        "path_planning_time", "path_execution_time", "path_smoothness_msj",
        "path_optimality_ratio", "collision_flag"]
print(df[cols].round(4).to_string(index=False))
