import numpy as np


def mean_squared_jerk(traj):
    """MSJ in joint space (rad^2/s^5), normalized by trajectory duration.
    Third-order forward finite difference per joint, summed across joints."""
    q = np.asarray(traj.waypoints, float)
    N = len(q)
    if N < 4:
        return float("nan")
    dt = traj.dt
    jerk = (q[3:] - 3 * q[2:-1] + 3 * q[1:-2] - q[:-3]) / dt**3   # (N-3, 7)
    sq = np.sum(jerk**2, axis=1)                                   # (N-3,)
    T = N * dt
    return float(np.sum(sq) * dt / T)
