import time
import numpy as np
from .core.types import Trajectory
from .planners.utils import retime_minjerk
from .metrics.smoothness import mean_squared_jerk
from .metrics.distance import path_length_ee, path_optimality_ratio

GOAL_TOL = 0.01
MAX_SPEED = 1.0
MIN_DURATION = 0.5


def execute(world, traj):
    collided, total = False, 0
    for q in traj.waypoints:
        n = world.contact_count(q)
        if n:
            collided = True
            total += n
    return collided, total, len(traj.waypoints) * traj.dt


def run_trial(world, planner, case, d_optimal, trial_id, sub_phase, trial_number, seed=None):
    start, goal = np.array(case["start"]), np.array(case["goal"])
    world.set_config(start)

    t0 = time.perf_counter()
    traj = planner.plan(world, start, goal)
    t_plan = time.perf_counter() - t0

    if d_optimal is None:
        d_optimal = traj.metadata.get("d_optimal")

    row = dict(trial_id=trial_id, sub_phase=sub_phase, algorithm=planner.name,
               test_case_id=case["test_case_id"], trial_number=trial_number, random_seed=seed,
               planning_success=bool(traj.planning_success),
               path_planning_time=t_plan, path_distance_optimal=d_optimal,
               algorithm_specific_metadata=traj.metadata)

    if not traj.planning_success:
        row.update(execution_success=False, path_execution_time=None, total_time=None,
                   path_smoothness_msj=None, path_distance_raw=None,
                   path_optimality_ratio=None, collision_flag=None, num_contact_points=None)
        world.reset_home()
        return row

    # common retiming only — no smoothing filter
    retimed = retime_minjerk(traj.waypoints, dt=world.dt,
                             max_speed=MAX_SPEED, min_duration=MIN_DURATION)
    timed = Trajectory(waypoints=retimed, dt=world.dt, planning_success=True,
                       metadata=traj.metadata)

    collided, n_contacts, t_exec = execute(world, timed)
    d_path = path_length_ee(world, timed)
    reached = float(np.max(np.abs(timed.waypoints[-1] - goal))) < GOAL_TOL

    row.update(execution_success=bool(reached),
               path_execution_time=t_exec, total_time=t_plan + t_exec,
               path_smoothness_msj=mean_squared_jerk(timed),
               path_distance_raw=d_path,
               path_optimality_ratio=path_optimality_ratio(d_path, d_optimal),
               collision_flag=bool(collided), num_contact_points=n_contacts)
    world.reset_home()
    return row
