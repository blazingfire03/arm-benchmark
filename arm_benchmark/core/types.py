from dataclasses import dataclass, field
import numpy as np


@dataclass
class Obstacle:
    obstacle_id: int
    shape_type: str                 # "box" | "cylinder" | "sphere"
    position: np.ndarray            # (3,) world xyz, meters
    dimensions: np.ndarray          # box: half-extents (hx,hy,hz); cyl: [r,h]; sphere: [r]
    orientation: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 1.]))
    pybullet_body_id: int = -1
    bounding_radius: float = 0.0


@dataclass
class Trajectory:
    waypoints: np.ndarray           # (N, 7) joint configs
    dt: float = 1.0 / 240.0
    planning_success: bool = True
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.waypoints) * self.dt
