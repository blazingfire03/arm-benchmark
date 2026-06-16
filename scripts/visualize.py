import sys, time, random, json, os
os.environ["NANOBIND_LEAK_WARNINGS"] = "0"

import numpy as np
import pybullet as p
from arm_benchmark.core.world import PandaWorld
from arm_benchmark.core.types import Trajectory
from arm_benchmark.scenes.obstacles import load_obstacles
from arm_benchmark.scenes.testcases import load_test_cases
from arm_benchmark.planners.min_jerk import MinJerk
from arm_benchmark.planners.astar import AStar
from arm_benchmark.planners.ompl_planners import OMPLPlanner
from arm_benchmark.planners.apf import APF
from arm_benchmark.planners.chomp import CHOMP
from arm_benchmark.planners.stomp import STOMP
from arm_benchmark.planners.utils import retime_minjerk
from arm_benchmark.metrics.smoothness import mean_squared_jerk
from arm_benchmark.metrics.distance import path_length_ee, path_optimality_ratio

OBSTACLES = "config/sparse_obstacles.json"
TESTCASES = "config/sparse_test_cases.json"
VIZ_MAX_SPEED = 0.3
VIZ_MIN_DURATION = 3.0
VIZ_JSON = "results/viz_results.json"

ALL_PLANNERS = {
    "min_jerk":     MinJerk(),
    "a_star":       AStar(),
    "RRT":          OMPLPlanner("RRT", time_budget=5.0),
    "RRT-Connect":  OMPLPlanner("RRT-Connect", time_budget=5.0),
    "RRT*":         OMPLPlanner("RRT*", time_budget=5.0),
    "PRM":          OMPLPlanner("PRM", time_budget=5.0),
    "APF":          APF(),
    "CHOMP":        CHOMP(),
    "STOMP":        STOMP(),
}

PLANNER = sys.argv[1] if len(sys.argv) > 1 else "min_jerk"
CASE_INDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 0

if PLANNER not in ALL_PLANNERS:
    print(f"unknown planner: {PLANNER}")
    print(f"available: {', '.join(ALL_PLANNERS.keys())}")
    sys.exit(1)

BLACK = [0, 0, 0]
GREEN = [0.1, 0.8, 0.2]
RED = [1.0, 0.1, 0.1]
YELLOW = [1.0, 0.85, 0.0]
BLUE = [0.1, 0.3, 1.0]

w = PandaWorld(gui=True, acm_samples=200)
cid = w.cid
top = float(w.base_pos[2])
p.resetDebugVisualizerCamera(1.8, 50, -30, [0, 0, top + 0.1], physicsClientId=cid)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)

load_obstacles(w, OBSTACLES)
cases = load_test_cases(TESTCASES)
case = cases[CASE_INDEX]
start, goal = np.array(case["start"]), np.array(case["goal"])

planner = ALL_PLANNERS[PLANNER]
print(f"planning with {PLANNER} on case {CASE_INDEX}...")
t0 = time.perf_counter()
traj = planner.plan(w, start, goal)
t_plan = time.perf_counter() - t0

# d_optimal
d_opt = traj.metadata.get("d_optimal")
if d_opt is None:
    a_traj = AStar().plan(w, start, goal)
    d_opt = a_traj.metadata.get("d_optimal") if a_traj.planning_success else None

# viz retiming (slow for animation)
viz_wps = retime_minjerk(traj.waypoints, dt=w.dt, max_speed=VIZ_MAX_SPEED,
                         min_duration=VIZ_MIN_DURATION)
timed = Trajectory(waypoints=viz_wps, dt=w.dt)

# benchmark retiming (for metrics)
bench_wps = retime_minjerk(traj.waypoints, dt=w.dt, max_speed=1.0, min_duration=0.5)
bench_traj = Trajectory(waypoints=bench_wps, dt=w.dt)

# walk viz path
ee_pts, hit = [], []
for q in timed.waypoints:
    ee_pts.append(w.ee_position(q))
    hit.append(w.contact_count(q) > 0)
ee_pts = np.array(ee_pts)
collided = any(hit)

# metrics on benchmark retiming
msj = mean_squared_jerk(bench_traj)
d_path = path_length_ee(w, bench_traj)
por = path_optimality_ratio(d_path, d_opt) if d_opt else float('nan')
t_exec = len(bench_traj.waypoints) * bench_traj.dt

# draw path
for i in range(len(ee_pts) - 1):
    color = RED if (hit[i] or hit[i + 1]) else GREEN
    p.addUserDebugLine(ee_pts[i].tolist(), ee_pts[i + 1].tolist(), color,
                       lineWidth=5.0, physicsClientId=cid)

# start ball (blue) and goal ball (yellow)
for pos, rgb in [(ee_pts[0], [0.1, 0.3, 1.0, 1]), (ee_pts[-1], [1.0, 0.9, 0.0, 1])]:
    vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=rgb, physicsClientId=cid)
    p.createMultiBody(0, -1, vs, basePosition=pos.tolist(), physicsClientId=cid)

# overlay
ok = traj.planning_success and not collided
overlay = [
    f"Algorithm: {PLANNER}",
    f"Case: {CASE_INDEX}",
    f"Status: {'SUCCESS' if ok else 'FAIL - collision' if collided else 'FAIL - no path'}",
    f"Plan time: {t_plan:.4f} s",
    f"Exec time: {t_exec:.3f} s",
    f"MSJ: {msj:.2f} rad^2/s^5",
    f"POR: {por:.4f}" if not np.isnan(por) else "POR: N/A",
    f"Path length: {d_path:.3f} m",
    f"d_optimal: {d_opt:.3f} m" if d_opt else "d_optimal: N/A",
    f"Collision: {'YES' if collided else 'NO'}",
]
for k, txt in enumerate(overlay):
    if k == 2:
        col = GREEN if ok else RED
    else:
        col = BLACK
    p.addUserDebugText(txt, [0.0, 0.0, top + 1.1 - 0.06 * k], col,
                       textSize=1.1, physicsClientId=cid)

# terminal output
print(f"\n{'='*50}")
print(f" {PLANNER.upper()} — Case {CASE_INDEX}")
print(f"{'='*50}")
print(f" Status:       {'SUCCESS' if ok else 'FAIL'}")
print(f" Plan time:    {t_plan:.4f} s")
print(f" Exec time:    {t_exec:.3f} s")
print(f" MSJ:          {msj:.2f} rad²/s⁵")
print(f" POR:          {por:.4f}" if not np.isnan(por) else " POR:          N/A")
print(f" Path length:  {d_path:.3f} m")
print(f" d_optimal:    {d_opt:.3f} m" if d_opt else " d_optimal:    N/A")
print(f" Collision:    {'YES' if collided else 'NO'}")
print(f"{'='*50}\n")

# animate
w.set_config(start)
time.sleep(0.3)
print(f"animating ({len(timed.waypoints)} waypoints, ~{timed.duration:.1f} s)...")
for q in timed.waypoints:
    w.set_config(q)
    p.stepSimulation(physicsClientId=cid)
    time.sleep(w.dt)

print("done — close window or Ctrl-C to exit.")
try:
    while True:
        p.stepSimulation(physicsClientId=cid)
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
w.disconnect()
