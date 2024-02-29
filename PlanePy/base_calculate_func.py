from PlanePy.util import BasePlane
from abc import ABC
from abc import abstractmethod


class BaseCalculateFunc(ABC):
    @staticmethod
    @abstractmethod
    def calculate_func(plane_x: BasePlane, planes_y: list, k: list, **kwargs):
        pass
