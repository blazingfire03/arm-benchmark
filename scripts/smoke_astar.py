import time
import numpy as np
from arm_benchmark.core.world import PandaWorld
from arm_benchmark.scenes.obstacles import load_obstacles
from arm_benchmark.scenes.testcases import load_test_cases
from arm_benchmark.planners.astar import AStar
from arm_benchmark.planners.min_jerk import MinJerk
from arm_benchmark.metrics.distance import path_length_ee, path_optimality_ratio

w = PandaWorld(gui=False)
load_obstacles(w, "config/sparse_obstacles.json")
cases = load_test_cases("config/sparse_test_cases.json")
c = cases[0]
start, goal = np.array(c["start"]), np.array(c["goal"])

print("running A* + shortcutting on case 0...")
t0 = time.perf_counter()
traj = AStar().plan(w, start, goal)
wall = time.perf_counter() - t0

print("success:", traj.planning_success, "| wall: %.2f s" % wall)
print("cells expanded:", traj.metadata.get("cells_expanded"),
      "| grid cells:", traj.metadata.get("grid_cells"),
      "-> shortcut corners:", traj.metadata.get("shortcut_corners"))
d_opt = traj.metadata["d_optimal"]
print("d_optimal (shortcut): %.3f m  (was 1.403 m pre-shortcut, straight-line %.3f m)"
      % (d_opt, c["straight_line_ee"]))

# A*'s own POR must be ~1.0 by construction
print("A* self-POR (sanity, must be ~1.0): %.4f" % path_optimality_ratio(path_length_ee(w, traj), d_opt))

# min-jerk POR should now be ~1.0 (no longer below 1)
mj = MinJerk().plan(w, start, goal)
print("min-jerk POR (was 0.814): %.4f" % path_optimality_ratio(path_length_ee(w, mj), d_opt))
w.disconnect()
