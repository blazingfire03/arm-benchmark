import os, sys, json
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

OUT = "results/visualizations"
os.makedirs(OUT, exist_ok=True)

ALL_PLANNERS = [
    ("min_jerk",    MinJerk(),                              [1.0, 0.2, 0.2]),
    ("a_star",      AStar(),                                [0.1, 0.8, 0.2]),
    ("RRT",         OMPLPlanner("RRT", time_budget=5.0),    [0.2, 0.4, 1.0]),
    ("RRT-Connect", OMPLPlanner("RRT-Connect", time_budget=5.0), [0.1, 0.8, 0.8]),
    ("RRT*",        OMPLPlanner("RRT*", time_budget=5.0),   [0.6, 0.3, 0.9]),
    ("PRM",         OMPLPlanner("PRM", time_budget=5.0),    [0.9, 0.7, 0.1]),
    ("APF",         APF(),                                  [0.9, 0.5, 0.1]),
    ("CHOMP",       CHOMP(),                                [0.1, 0.5, 0.8]),
    ("STOMP",       STOMP(),                                [0.8, 0.2, 0.5]),
]

SUB_PHASES = ["sparse", "moderate", "dense"]
N_CASES = 5
WIDTH, HEIGHT = 1280, 960

def capture_screenshot(cid, width=WIDTH, height=HEIGHT):
    _, _, rgb, _, _ = p.getCameraImage(width, height, physicsClientId=cid,
        renderer=p.ER_TINY_RENDERER)
    return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def draw_path(world, traj, color, line_width=4.0):
    cid = world.cid
    pts = [world.ee_position(q) for q in traj.waypoints[::3]]  # subsample for speed
    for i in range(len(pts) - 1):
        p.addUserDebugLine(pts[i].tolist(), pts[i+1].tolist(), color,
                           lineWidth=line_width, physicsClientId=cid)


def draw_markers(world, start_ee, goal_ee):
    cid = world.cid
    for pos, rgb in [(start_ee, [0.1, 0.3, 1.0, 1]), (goal_ee, [1.0, 0.9, 0.0, 1])]:
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=rgb, physicsClientId=cid)
        p.createMultiBody(0, -1, vs, basePosition=pos.tolist(), physicsClientId=cid)


print("generating visualizations for all sub-phases, algorithms, and cases...\n")

for sp in SUB_PHASES:
    obs_file = f"config/{sp}_obstacles.json"
    cases_file = f"config/{sp}_test_cases.json"
    cases = load_test_cases(cases_file)[:N_CASES]

    print(f"--- {sp.upper()} ---")

    # per-algorithm per-case screenshots
    for ci, case in enumerate(cases):
        start, goal = np.array(case["start"]), np.array(case["goal"])

        for algo_name, planner, color in ALL_PLANNERS:
            w = PandaWorld(gui=False, acm_samples=100)
            load_obstacles(w, obs_file)
            top = float(w.base_pos[2])
            p.resetDebugVisualizerCamera(1.8, 50, -30, [0, 0, top + 0.1], physicsClientId=w.cid)

            traj = planner.plan(w, start, goal)
            if not traj.planning_success:
                w.disconnect()
                continue

            retimed = retime_minjerk(traj.waypoints, dt=w.dt, max_speed=0.5, min_duration=2.0)
            timed = Trajectory(waypoints=retimed, dt=w.dt)

            # metrics
            bench = Trajectory(waypoints=retime_minjerk(traj.waypoints, dt=w.dt), dt=w.dt)
            msj = mean_squared_jerk(bench)
            d_path = path_length_ee(w, bench)

            # check collision
            collided = any(w.contact_count(q) > 0 for q in timed.waypoints[::5])

            # draw
            path_color = [1.0, 0.1, 0.1] if collided else color
            draw_path(w, timed, path_color)
            start_ee = w.ee_position(start)
            goal_ee = w.ee_position(goal)
            draw_markers(w, start_ee, goal_ee)

            # overlay text
            status = "COLLISION" if collided else "OK"
            p.addUserDebugText(f"{algo_name}", [0, 0, top + 0.95], [0, 0, 0],
                               textSize=1.5, physicsClientId=w.cid)
            p.addUserDebugText(f"MSJ: {msj:.1f}  |  {status}",
                               [0, 0, top + 0.88], [0, 0, 0],
                               textSize=1.0, physicsClientId=w.cid)

            w.set_config(start)
            img = capture_screenshot(w.cid)
            fname = f"{OUT}/{sp}_case{ci}_{algo_name}.png"
            Image.fromarray(img).save(fname)
            w.disconnect()

        print(f"  case {ci}: all algorithms rendered")

    # --- combined grid per sub-phase: all algorithms on case 0 ---
    case = cases[0]
    start, goal = np.array(case["start"]), np.array(case["goal"])

    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 5, figure=fig, wspace=0.05, hspace=0.15)

    for idx, (algo_name, planner, color) in enumerate(ALL_PLANNERS):
        fname = f"{OUT}/{sp}_case0_{algo_name}.png"
        if not os.path.exists(fname):
            continue
        img = Image.open(fname)
        row, col = idx // 5, idx % 5
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(algo_name, fontsize=14, fontweight='bold')
        ax.axis('off')

    fig.suptitle(f"Phase 1 — {sp.upper()} Sub-phase (Case 0)", fontsize=18, fontweight='bold')
    grid_path = f"{OUT}/grid_{sp}.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  grid saved: {grid_path}")

print(f"\n{'='*60}")
print(f"  DONE — {OUT}/")
n_files = len([f for f in os.listdir(OUT) if f.endswith('.png')])
print(f"  {n_files} images generated")
print(f"  grid_{'{sp}'}.png = all algorithms side-by-side")
print(f"  {'{sp}'}_case{'{N}'}_{'{algo}'}.png = individual shots")
print(f"{'='*60}")
