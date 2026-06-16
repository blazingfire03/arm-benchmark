import numpy as np


def path_length_ee(world, traj):
    """Total end-effector Cartesian path length (m) along the trajectory."""
    pts = np.array([world.ee_position(q) for q in traj.waypoints])
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def path_optimality_ratio(d_path, d_optimal):
    """POR = d_path / d_optimal. d_optimal comes from A* (None until step 4b)."""
    if not d_optimal or d_optimal <= 0:
        return float("nan")
    return d_path / d_optimal
