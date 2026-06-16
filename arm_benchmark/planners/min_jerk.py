import numpy as np
from .base import Planner
from ..core.types import Trajectory

MAX_JOINT_SPEED = 1.0   # rad/s, nominal — PLACEHOLDER, sets baseline duration/MSJ
MIN_DURATION = 0.5      # s


class MinJerk(Planner):
    name = "min_jerk"

    def __init__(self, dt=1/240, max_speed=MAX_JOINT_SPEED, min_duration=MIN_DURATION):
        self.dt = dt
        self.max_speed = max_speed
        self.min_duration = min_duration

    def plan(self, world, start, goal) -> Trajectory:
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        disp = np.abs(goal - start)
        T = max(self.min_duration, float(disp.max()) / self.max_speed)
        N = max(4, int(round(T / self.dt)))
        s = np.linspace(0.0, 1.0, N)[:, None]
        profile = 10 * s**3 - 15 * s**4 + 6 * s**5      # min-jerk: 0 vel/accel at ends
        waypoints = start[None, :] + (goal - start)[None, :] * profile
        return Trajectory(waypoints=waypoints, dt=self.dt, planning_success=True,
                          metadata={"duration": T, "n_samples": N})
