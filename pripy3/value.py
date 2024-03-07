import numpy as np
from sko.PSO import PSO

from pripy3.base_calculate_func import BaseCalculateFunc
from pripy3.util import BasePlane
from pripy3.util import single_priority


class Value:
    def __init__(self, func: BaseCalculateFunc = None, **kwargs_value):
        self.kwargs_value = kwargs_value
        if func is None:
            class PsoCalculateFunc(BaseCalculateFunc):
                @staticmethod
                def calculate_func(plane_x: BasePlane, planes_y: list, **kwargs) -> float:
                    # Set default values for kwargs
                    kwargs.setdefault('k', [0.1, 0.3, 0.6])
                    kwargs.setdefault('time', 0.1)
                    kwargs.setdefault('pop_size', 50)
                    kwargs.setdefault('max_iter', 100)
                    kwargs.setdefault('w', 0.8)
                    kwargs.setdefault('c1', 0.5)
                    kwargs.setdefault('c2', 0.5)

                    way = kwargs['way']
                    k_val = kwargs['k']
                    time_val = kwargs['time']
                    pop_size_val = kwargs['pop_size']
                    max_iter_val = kwargs['max_iter']
                    w_val = kwargs['w']
                    c1_val = kwargs['c1']
                    c2_val = kwargs['c2']

                    def cost_func(x):
                        x = x[0]
                        updated_plane = plane_x.calculate_updated_plane(
                            values=tuple(((x if i == way else 0) for i in range(3))), time=time_val)

                        return sum(
                            -single_priority(plane_x=updated_plane, plane_y=plane_y, k=k_val) for plane_y in planes_y)

                    pso = PSO(func=cost_func, n_dim=1, pop=pop_size_val, max_iter=max_iter_val,
                              ub=[plane_x.ubs[way]], lb=[-plane_x.ubs[way]], w=w_val, c1=c1_val, c2=c2_val)

                    pso.run()
                    return float(pso.gbest_x[0])

            func = PsoCalculateFunc.calculate_func

        self.__value = np.zeros(shape=3, dtype=float)
        self.__func = func

    def calculate(self, plane_x: BasePlane, planes_y: list, time: float = 0.1):
        for way in range(3):
            self.__value[way] = self.__func(way=way, plane_x=plane_x, planes_y=planes_y, time=time, **self.kwargs_value)

    @property
    def value(self):
        return self.__value
