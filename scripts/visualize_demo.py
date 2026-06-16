import os, time
os.environ["NANOBIND_LEAK_WARNINGS"] = "0"

import numpy as np
import pybullet as p
from arm_benchmark.core.world import PandaWorld
from arm_benchmark.core.types import Trajectory
from arm_benchmark.scenes.obstacles import generate_obstacles
from arm_benchmark.scenes.testcases import generate_test_cases
from arm_benchmark.planners.min_jerk import MinJerk
from arm_benchmark.planners.astar import AStar
from arm_benchmark.planners.ompl_planners import OMPLPlanner
from arm_benchmark.planners.apf import APF
from arm_benchmark.planners.chomp import CHOMP
from arm_benchmark.planners.stomp import STOMP
from arm_benchmark.planners.utils import retime_minjerk
from arm_benchmark.metrics.smoothness import mean_squared_jerk
from arm_benchmark.metrics.distance import path_length_ee

N_OBSTACLES = 8
OBS_SEED = 42
CASE_INDEX = 0
WAIT = 6.0

ALL_PLANNERS = [
    ("min_jerk",    MinJerk()),
    ("a_star",      AStar()),
    ("RRT",         OMPLPlanner("RRT", time_budget=5.0)),
    ("RRT-Connect", OMPLPlanner("RRT-Connect", time_budget=5.0)),
    ("RRT*",        OMPLPlanner("RRT*", time_budget=5.0)),
    ("PRM",         OMPLPlanner("PRM", time_budget=5.0)),
    ("APF",         APF()),
    ("CHOMP",       CHOMP()),
    ("STOMP",       STOMP()),
]

GREEN, RED, BLACK = [0.1,0.8,0.2], [1.0,0.1,0.1], [0,0,0]

# generate scene once (headless) to get obstacle + case data
print("generating demo scene...")
w_tmp = PandaWorld(gui=False, acm_samples=100)
obs_list = generate_obstacles(w_tmp, N_OBSTACLES, seed=OBS_SEED)
cases = generate_test_cases(w_tmp, n=10, seed=OBS_SEED + 100)
# save obstacle positions for reload
obs_data = []
for o in obs_list:
    obs_data.append(dict(obstacle_id=o.obstacle_id, shape_type=o.shape_type,
                         position=o.position.tolist(), dimensions=o.dimensions.tolist(),
                         orientation=o.orientation.tolist(), bounding_radius=o.bounding_radius))
w_tmp.disconnect()

import json
os.makedirs("config", exist_ok=True)
with open("config/demo_obstacles.json", "w") as f:
    json.dump(obs_data, f, indent=2)

case = cases[CASE_INDEX]
start, goal = np.array(case["start"]), np.array(case["goal"])
total = len(ALL_PLANNERS)

from arm_benchmark.scenes.obstacles import load_obstacles

def hold(cid, sec):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        try:
            p.stepSimulation(physicsClientId=cid)
        except:
            return
        time.sleep(0.016)

print(f"DEMO ENVIRONMENT — {N_OBSTACLES} obstacles, case {CASE_INDEX}")
print(f"running {total} algorithms\n")

for i, (name, planner) in enumerate(ALL_PLANNERS):
    print(f"\n[{i+1}/{total}] DEMO — {name}")
    w = PandaWorld(gui=True, acm_samples=100)
    cid = w.cid
    top = float(w.base_pos[2])
    p.resetDebugVisualizerCamera(1.8, 50, -30, [0, 0, top+0.1], physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    load_obstacles(w, "config/demo_obstacles.json")

    t0 = time.perf_counter()
    traj = planner.plan(w, start, goal)
    t_plan = time.perf_counter() - t0

    if not traj.planning_success:
        p.addUserDebugText(f"DEMO - {name}", [0,0,top+1.0], BLACK, textSize=1.5, physicsClientId=cid)
        p.addUserDebugText("PLANNING FAILED", [0,0,top+0.92], RED, textSize=1.3, physicsClientId=cid)
        meta = traj.metadata if isinstance(traj.metadata, dict) else {}
        p.addUserDebugText(f"Reason: {meta.get('reason','?')}", [0,0,top+0.84], BLACK, textSize=1.0, physicsClientId=cid)
        w.set_config(start)
        print(f"  FAILED: {meta.get('reason','?')}")
        hold(cid, WAIT)
        w.disconnect()
        continue

    viz_wps = retime_minjerk(traj.waypoints, dt=w.dt, max_speed=0.3, min_duration=3.0)
    timed = Trajectory(waypoints=viz_wps, dt=w.dt)
    bench_wps = retime_minjerk(traj.waypoints, dt=w.dt, max_speed=1.0, min_duration=0.5)
    bench_traj = Trajectory(waypoints=bench_wps, dt=w.dt)
    msj = mean_squared_jerk(bench_traj)
    d_path = path_length_ee(w, bench_traj)
    t_exec = len(bench_traj.waypoints) * bench_traj.dt

    step = max(1, len(timed.waypoints) // 25)
    idx = list(range(0, len(timed.waypoints), step))
    if idx[-1] != len(timed.waypoints)-1:
        idx.append(len(timed.waypoints)-1)

    ee, hit_any = [], False
    for si in idx:
        ee.append(w.ee_position(timed.waypoints[si]))
        if w.contact_count(timed.waypoints[si]) > 0:
            hit_any = True

    for j in range(len(ee)-1):
        p.addUserDebugLine(ee[j].tolist(), ee[j+1].tolist(), RED if hit_any else GREEN, lineWidth=4.0, physicsClientId=cid)

    for pos, rgba in [(ee[0],[0.1,0.3,1.0,1]), (ee[-1],[1.0,0.9,0.0,1])]:
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=rgba, physicsClientId=cid)
        p.createMultiBody(0,-1,vs, basePosition=pos.tolist(), physicsClientId=cid)

    ok = not hit_any
    p.addUserDebugText(f"DEMO - {name}", [0,0,top+1.0], BLACK, textSize=1.5, physicsClientId=cid)
    p.addUserDebugText(f"{'SUCCESS' if ok else 'COLLISION'}", [0,0,top+0.92], GREEN if ok else RED, textSize=1.3, physicsClientId=cid)
    p.addUserDebugText(f"Plan: {t_plan:.3f}s  MSJ: {msj:.1f}", [0,0,top+0.84], BLACK, textSize=1.0, physicsClientId=cid)
    p.addUserDebugText(f"Path: {d_path:.3f}m  Exec: {t_exec:.3f}s", [0,0,top+0.76], BLACK, textSize=1.0, physicsClientId=cid)

    print(f"  {'OK' if ok else 'COLLISION'}  plan={t_plan:.3f}s  MSJ={msj:.1f}")

    w.set_config(start)
    time.sleep(0.2)
    for q in timed.waypoints:
        w.set_config(q)
        p.stepSimulation(physicsClientId=cid)
        time.sleep(w.dt)

    hold(cid, WAIT)
    w.disconnect()

print(f"\n  DEMO — ALL {total} ALGORITHMS COMPLETE")
