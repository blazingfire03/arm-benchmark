import numpy as np
from arm_benchmark.core.world import PandaWorld
from arm_benchmark.core.types import Trajectory
from arm_benchmark.scenes.obstacles import load_obstacles
from arm_benchmark.scenes.testcases import load_test_cases
from arm_benchmark.planners.min_jerk import MinJerk
from arm_benchmark.metrics.smoothness import mean_squared_jerk
from arm_benchmark.metrics.distance import path_length_ee

w = PandaWorld(gui=False)
load_obstacles(w, "config/sparse_obstacles.json")
cases = load_test_cases("config/sparse_test_cases.json")

c = cases[0]
start, goal = np.array(c["start"]), np.array(c["goal"])
traj = MinJerk().plan(w, start, goal)
print("waypoints:", traj.waypoints.shape, "| duration: %.3f s" % traj.duration)
print("starts at start:", np.allclose(traj.waypoints[0], start),
      "| ends at goal:", np.allclose(traj.waypoints[-1], goal))

msj = mean_squared_jerk(traj)
dlen = path_length_ee(w, traj)
print("min-jerk MSJ:  %.4e" % msj)
print("EE path length: %.3f m  (straight-line %.3f m)" % (dlen, c["straight_line_ee"]))

rng = np.random.default_rng(0)
jerky = Trajectory(waypoints=traj.waypoints + 0.05 * rng.standard_normal(traj.waypoints.shape),
                   dt=traj.dt)
print("jerky MSJ:     %.4e  (must be >> min-jerk)" % mean_squared_jerk(jerky))
w.disconnect()
