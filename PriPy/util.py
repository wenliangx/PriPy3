from typing import Union

import numpy as np

from PlanePy.util import BasePlane
from PlanePy.util import translate_velocity


def priority_d(position_x: np.ndarray,
               position_y: np.ndarray,
               r_min: Union[int, float] = 10,
               r_max: Union[int, float] = 30):
    return float(
        np.exp(-np.power((np.linalg.norm(position_y - position_x) - (r_min + r_max) / 2) / (r_max - r_min), 2)))


def priority_v(velocity_x: np.ndarray, velocity_y: np.ndarray):
    v_x = velocity_x[0]
    v_y = velocity_y[0]
    if v_x < 0.6 * v_y:
        return 0.1
    elif v_x < 1.5 * v_y:
        return float(v_x) / float(v_y) - 0.5
    else:
        return 1


def priority_a(plane_x: BasePlane, plane_y: BasePlane):
    velocity_x, velocity_y = translate_velocity(plane_x.velocity), translate_velocity(plane_y.velocity)  # 转化为直角坐标系下计算
    if np.linalg.norm(velocity_x) == 0 or np.linalg.norm(velocity_y) == 0:
        return -10
    position = plane_y.position - plane_x.position
    a_i, a_v = (np.dot(position, velocity_y) / (np.linalg.norm(position) * np.linalg.norm(velocity_y)),
                np.dot(position, velocity_x) / (np.linalg.norm(position) * np.linalg.norm(velocity_x)))
    a_i, a_v = np.arccos(a_i), np.arccos(a_v)
    return 1 - (float(a_i) + float(a_v)) / (2 * np.pi)


def single_priority(plane_x: BasePlane, plane_y: BasePlane, k: list):
    total_priority = np.array(object=[priority_d(position_x=plane_x.position, position_y=plane_y.position),
                                      priority_v(velocity_x=plane_x.velocity, velocity_y=plane_y.velocity),
                                      priority_a(plane_x=plane_x, plane_y=plane_y)],
                              dtype=np.float64)
    return np.dot(total_priority, np.array(k, dtype=np.float64))
