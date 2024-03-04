from typing import Union

import numpy as np

from PlanePy.util import BasePlane
from PlanePy.value import Value


# 比起基础飞机，添加计算最佳策略的类Value
class Plane(BasePlane):
    def __init__(self, position: Union[list, tuple, np.ndarray] = None,
                 velocity: Union[list, tuple, np.ndarray] = None,
                 velocity_limit: float = 10.0, ubs: tuple = None,
                 func=None, **kwargs_in_func):
        super(Plane, self).__init__(position, velocity, velocity_limit, ubs)
        self.__values = Value(func=func, **kwargs_in_func)

    # 通过输入值（速度改变量）和时间间隔计算并返回下一时刻的飞机
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

    # 返回父类基础飞机
    @property
    def base_plane(self) -> BasePlane:
        return BasePlane(self.position, self.velocity, self.velocity_limit, self.ubs)

    # 计算最佳策略值
    def calculate_values(self, planes_y: list = None, time: float = 0.1, **kwargs):
        self.__values.calculate(plane_x=self.base_plane, planes_y=planes_y, time=time)

    def __del__(self):
        super(Plane, self).__del__()
        del self.__values
