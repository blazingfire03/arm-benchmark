from abc import ABC, abstractmethod
from ..core.types import Trajectory


class Planner(ABC):
    name = "base"

    @abstractmethod
    def plan(self, world, start, goal) -> Trajectory:
        """Return a Trajectory. Set planning_success=False on failure/timeout."""
        ...
