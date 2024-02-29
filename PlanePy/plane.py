import numpy as np
from PlanePy.util import BasePlane
from typing import Union
from PlanePy.value import Value


class Plane(BasePlane):
    def __init__(self, position: Union[list, tuple, np.ndarray] = None,
                 velocity: Union[list, tuple, np.ndarray] = None,
                 velocity_limit: float = 100.0, ubs: tuple = None,
                 func=None):
        super(Plane, self).__init__(position, velocity, velocity_limit, ubs)
        self.__values = Value(func=func)

    def calculate_updated_plane(self, values: Union[list, tuple, np.ndarray], time: float):

        updated_base_plane = super().calculate_updated_plane(values=values, time=time)
        updated_plane = Plane(position=updated_base_plane.position,
                              velocity=updated_base_plane.velocity,
                              velocity_limit=updated_base_plane.velocity_limit,
                              ubs=updated_base_plane.ubs)
        return updated_plane

    @property
    def values(self):
        return self.__values.value

    @property
    def base_plane(self):
        return BasePlane(position=self.position, velocity=self.velocity,
                         velocity_limit=self.velocity_limit, ubs=self.ubs)

    def calculate_values(self, planes_y: list = None, k: list = None, **kwargs):
        self.__values.calculate(plane_x=self.base_plane, planes_y=planes_y, k=k, **kwargs)

    def __del__(self):
        del self.position
        del self.velocity
