from abc import ABC
from abc import abstractmethod

from numpy import ndarray

from PlanePy.util import BasePlane


class BaseCalculateFunc(ABC):
    @staticmethod
    @abstractmethod
    def calculate_func(plane_x: BasePlane, planes_y: list, **kwargs) -> ndarray:
        pass
