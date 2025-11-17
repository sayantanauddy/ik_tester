import numpy as np
from abc import ABC, abstractmethod

class IKSolver(ABC):
    """
    Abstract base class for numerical IK solvers.
    Enforces the function signature:
        solve(q_pos, p_des, quat_des, q_ref) -> np.ndarray
    """

    @abstractmethod
    def solve(
        self,
        q_pos: np.ndarray,
        p_des: np.ndarray,
        quat_des: np.ndarray,
        q_ref: np.ndarray,
    ) -> np.ndarray:
        pass
