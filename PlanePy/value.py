import numpy as np
from PlanePy.util import BasePlane
from PriPy.util import single_priority
from sko.PSO import PSO
from PlanePy.base_calculate_func import BaseCalculateFunc


class Value:
    def __init__(self, func: BaseCalculateFunc = None):
        if func is None:
            class PsoCalculateFunc(BaseCalculateFunc):
                @staticmethod
                def calculate_func(plane_x: BasePlane, planes_y: list, k: list = None, **kwargs) -> float:
                    if k is None:
                        k = [0.1, 0.3, 0.6]

                    def cost_func(x):
                        x = x[0]
                        updated_plane = plane_x.calculate_updated_plane(
                            values=tuple(((x if i == kwargs['way'] else 0) for i in range(3))), time=kwargs['time'])
                        # print(list(map(
                        #     lambda plane_y: -single_priority(plane_x=updated_plane, plane_y=plane_y, k=k), planes_y
                        # )))
                        return sum(list(map(
                            lambda plane_y: -single_priority(plane_x=updated_plane, plane_y=plane_y, k=k), planes_y
                        )))
                    if kwargs.get('size_pop') is None:
                        kwargs['size_pop'] = 50
                    if kwargs.get('max_iter') is None:
                        kwargs['max_iter'] = 100
                    if kwargs.get('w') is None:
                        kwargs['w'] = 0.8
                    if kwargs.get('c1') is None:
                        kwargs['c1'] = 0.5
                    if kwargs.get('c2') is None:
                        kwargs['c2'] = 0.5

                    # 粒子群算法计算值
                    pso = PSO(func=cost_func, n_dim=1, pop=kwargs['size_pop'], max_iter=kwargs['max_iter'],
                              ub=[plane_x.ubs[kwargs['way']]], lb=[-plane_x.ubs[kwargs['way']]],
                              w=kwargs['w'], c1=kwargs['c1'], c2=kwargs['c2'])
                    pso.run()
                    return float(pso.gbest_x)
            func = PsoCalculateFunc.calculate_func

        self.__value = np.zeros(shape=3, dtype=float)
        self.__func = func

    def calculate(self, plane_x: BasePlane, planes_y: list, k: list = None, **kwargs):
        for way in range(3):
            self.__value[way] = self.__func(way=way, plane_x=plane_x, planes_y=planes_y, k=k, **kwargs)

    @property
    def value(self):
        return self.__value
