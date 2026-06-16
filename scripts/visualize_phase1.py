import os, sys, time, select, threading
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
from arm_benchmark.metrics.distance import path_length_ee

CASE_INDEX = 2
VIZ_MAX_SPEED = 0.3
VIZ_MIN_DURATION = 3.0
WAIT_AFTER_ANIM = 5.0   # seconds to hold each viz after animation

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

SUB_PHASES = ["sparse", "moderate", "dense"]
GREEN = [0.1, 0.8, 0.2]
RED = [1.0, 0.1, 0.1]
BLACK = [0, 0, 0]

total = len(SUB_PHASES) * len(ALL_PLANNERS)
count = 0

def wait_with_sim(cid, seconds):
    """Keep stepping simulation while waiting — prevents X11 timeout."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        try:
            p.stepSimulation(physicsClientId=cid)
        except:
            return
        time.sleep(0.016)  # ~60fps

for sp in SUB_PHASES:
    obs_file = f"config/{sp}_obstacles.json"
    cases_file = f"config/{sp}_test_cases.json"
    cases = load_test_cases(cases_file)
    case = cases[CASE_INDEX]
    start, goal = np.array(case["start"]), np.array(case["goal"])

    for algo_name, planner in ALL_PLANNERS:
        count += 1
        print(f"\n[{count}/{total}] {sp.upper()} — {algo_name}")

        w = PandaWorld(gui=True, acm_samples=100)
        cid = w.cid
        top = float(w.base_pos[2])
        p.resetDebugVisualizerCamera(1.8, 50, -30, [0, 0, top + 0.1], physicsClientId=cid)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
        load_obstacles(w, obs_file)

        t0 = time.perf_counter()
        traj = planner.plan(w, start, goal)
        t_plan = time.perf_counter() - t0

        if not traj.planning_success:
            p.addUserDebugText(f"{sp.upper()} - {algo_name}", [0, 0, top+1.0], BLACK,
                               textSize=1.5, physicsClientId=cid)
            p.addUserDebugText("PLANNING FAILED", [0, 0, top+0.92], RED,
                               textSize=1.3, physicsClientId=cid)
            meta = traj.metadata if isinstance(traj.metadata, dict) else {}
            p.addUserDebugText(f"Reason: {meta.get('reason','?')}", [0, 0, top+0.84], BLACK,
                               textSize=1.0, physicsClientId=cid)
            w.set_config(start)
            print(f"  FAILED: {meta.get('reason','?')}")
            print(f"  showing for {WAIT_AFTER_ANIM}s...")
            wait_with_sim(cid, WAIT_AFTER_ANIM)
            w.disconnect()
            continue

        viz_wps = retime_minjerk(traj.waypoints, dt=w.dt, max_speed=VIZ_MAX_SPEED,
                                 min_duration=VIZ_MIN_DURATION)
        timed = Trajectory(waypoints=viz_wps, dt=w.dt)

        bench_wps = retime_minjerk(traj.waypoints, dt=w.dt, max_speed=1.0, min_duration=0.5)
        bench_traj = Trajectory(waypoints=bench_wps, dt=w.dt)
        msj = mean_squared_jerk(bench_traj)
        d_path = path_length_ee(w, bench_traj)
        t_exec = len(bench_traj.waypoints) * bench_traj.dt

        n_pts = len(timed.waypoints)
        step = max(1, n_pts // 25)
        sample_idx = list(range(0, n_pts, step))
        if sample_idx[-1] != n_pts - 1:
            sample_idx.append(n_pts - 1)

        ee_pts = []
        hit_any = False
        for si in sample_idx:
            q = timed.waypoints[si]
            ee_pts.append(w.ee_position(q))
            if w.contact_count(q) > 0:
                hit_any = True

        for i in range(len(ee_pts) - 1):
            color = RED if hit_any else GREEN
            p.addUserDebugLine(ee_pts[i].tolist(), ee_pts[i+1].tolist(), color,
                               lineWidth=4.0, physicsClientId=cid)

        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[0.1,0.3,1.0,1],
                                 physicsClientId=cid)
        p.createMultiBody(0, -1, vs, basePosition=ee_pts[0].tolist(), physicsClientId=cid)
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[1.0,0.9,0.0,1],
                                 physicsClientId=cid)
        p.createMultiBody(0, -1, vs, basePosition=ee_pts[-1].tolist(), physicsClientId=cid)

        ok = not hit_any
        p.addUserDebugText(f"{sp.upper()} - {algo_name}", [0, 0, top+1.0], BLACK,
                           textSize=1.5, physicsClientId=cid)
        p.addUserDebugText(f"{'SUCCESS' if ok else 'COLLISION'}", [0, 0, top+0.92],
                           GREEN if ok else RED, textSize=1.3, physicsClientId=cid)
        p.addUserDebugText(f"Plan: {t_plan:.3f}s  MSJ: {msj:.1f}", [0, 0, top+0.84],
                           BLACK, textSize=1.0, physicsClientId=cid)
        p.addUserDebugText(f"Path: {d_path:.3f}m  Exec: {t_exec:.3f}s", [0, 0, top+0.76],
                           BLACK, textSize=1.0, physicsClientId=cid)

        print(f"  {'OK' if ok else 'COLLISION'}  plan={t_plan:.3f}s  MSJ={msj:.1f}")

        # animate
        w.set_config(start)
        time.sleep(0.2)
        for q in timed.waypoints:
            w.set_config(q)
            p.stepSimulation(physicsClientId=cid)
            time.sleep(w.dt)

        # hold view while keeping sim alive
        print(f"  holding for {WAIT_AFTER_ANIM}s...")
        wait_with_sim(cid, WAIT_AFTER_ANIM)
        w.disconnect()

print(f"\n{'='*50}")
print(f"  ALL {total} VISUALIZATIONS COMPLETE")
print(f"{'='*50}")
