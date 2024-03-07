from typing import Union

import numpy as np
import copy


class BasePlane:
    def __init__(self,
                 position: Union[list, tuple, np.ndarray] = None,
                 velocity: Union[list, tuple, np.ndarray] = None,
                 velocity_limit: float = 10.0, ubs: tuple = None):
        # Initialize position
        self.position = np.array(position, dtype=np.float64) if position is not None else (
            np.array([0, 0, 0], dtype=np.float64))

        # Initialize velocity
        self.velocity = np.array(velocity, dtype=np.float64) if velocity is not None else (
            np.array([5, 0, np.pi / 2], dtype=np.float64))

        # Regularize velocity
        regularization(self.velocity)

        # Set velocity limit
        self.velocity_limit = velocity_limit

        # Set upper bounds
        self.ubs = ubs if ubs is not None else (2, np.pi / 20, np.pi / 24)

    # Calculate and return the updated plane for the next time step
    def calculate_updated_plane(self, values: Union[list, tuple, np.ndarray], time: float):
        values = np.array(values)
        updated_plane = copy.deepcopy(self)
        updated_plane.velocity += values
        updated_plane.velocity[0] = min(float(updated_plane.velocity[0]), self.velocity_limit)
        regularization(updated_plane.velocity)
        updated_plane.position += 0.5 * time * (transform_velocity(updated_plane.velocity) + self.velocity)
        return updated_plane


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
    velocity_x, velocity_y = transform_velocity(plane_x.velocity), transform_velocity(plane_y.velocity)  # 转化为直角坐标系下计算
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


def base3_transform(num, size):
    result = [0] * size
    for i in range(size):
        result[i] = num % 3
        num //= 3
    return result


# 将速度（球坐标系）正规化

def regularization(velocity: np.ndarray) -> np.ndarray:
    # Adjust velocity magnitude to be positive
    if velocity[0] < 0:
        velocity[0] = -velocity[0]
        velocity[1] += np.pi
        velocity[2] += np.pi

    # Ensure elevation angle is between 0 and PI
    if velocity[2] < 0:
        velocity[2] = -velocity[2]
        velocity[1] += np.pi
    velocity[2] %= (2 * np.pi)  # Using modulo operator to keep angle within range

    # Ensure azimuth angle is between 0 and 2*PI
    velocity[1] %= (2 * np.pi)  # Using modulo operator to keep angle within range

    return velocity


# 将球坐标系的速度转换到直角坐标系中
def transform_velocity(velocity: np.ndarray) -> np.ndarray:
    regularization(velocity)
    return velocity[0] * np.array(object=[np.cos(velocity[1]) * np.sin(velocity[2]),
                                          np.sin(velocity[1]) * np.sin(velocity[2]),
                                          np.cos(velocity[2])], dtype=np.float64)
