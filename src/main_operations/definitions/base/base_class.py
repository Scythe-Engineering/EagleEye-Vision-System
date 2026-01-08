from typing import Any
from abc import ABC, abstractmethod
import numpy as np


class OperationInstance(ABC):
    """
    Base class for secondary operations
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Abstract method for run, all secondary operations must have this
        """
        pass

    def visualize(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Optional method for visualization
        """
        pass

    def update_config(self, json_config: dict[str, Any]) -> None:
        """
        Optional method for updating config during runtime
        """
        pass
