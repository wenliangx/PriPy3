import numpy as np
from typing import Union


# 将速度（柱坐标系）正规化
def regularization(velocity: np.ndarray) -> np.ndarray:
    # 速率调整为正数
    if velocity[0] < 0:
        velocity[0] = -velocity[0]
        velocity[1] = velocity[1] + np.pi
        velocity[2] = velocity[2] + np.pi

    # 仰角调整在0到PI之间
    if velocity[2] < 0:
        velocity[2] = -velocity[2]
        velocity[1] = velocity[1] + np.pi
    while velocity[2] > np.pi:
        velocity[2] = velocity[2] - np.pi
        velocity[1] = velocity[1] + np.pi
        
    # 方位角调整到0到2*PI之间
    while velocity[1] < 0:
        velocity[1] = velocity[1] + 2 * np.pi
    while velocity[1] > 2 * np.pi:
        velocity[1] = velocity[1] - 2 * np.pi
    return velocity


# 将球坐标系的速度转换到直角坐标系中
def translate_velocity(velocity: np.ndarray) -> np.ndarray:
    velocity = regularization(velocity)
    return velocity[0] * np.array(object=[np.cos(velocity[1]) * np.sin(velocity[2]),
                                          np.sin(velocity[1]) * np.sin(velocity[2]),
                                          np.cos(velocity[2])], dtype=np.float64)


# 计算下一时刻的位置
def calculate_update_position(position: np.ndarray, velocity: np.ndarray, time: float = 0.5) -> np.ndarray:
    return position + time * translate_velocity(velocity)


# 计算下一时刻的速度
def calculate_update_velocity(velocity: np.ndarray, way_calculate_velocity: int,
                              value: float = 0.0, velocity_limit: float = 100.0) -> np.ndarray:
    if way_calculate_velocity == 0:
        velocity[0] = min(float(value) + float(velocity[0]), velocity_limit)
    elif way_calculate_velocity == 1:
        velocity[1] = velocity[1] - value
    elif way_calculate_velocity == 2:
        velocity[2] = velocity[2] - value
    velocity = regularization(velocity)
    return velocity


# 基础飞机，包含位置，速度，速度上限和加速度上限
class BasePlane:
    def __init__(self, position: Union[list, tuple, np.ndarray] = None,
                 velocity: Union[list, tuple, np.ndarray] = None,
                 velocity_limit: float = 100.0, ubs: tuple = None, ):

        if position is None:
            self.position = np.array(object=[0, 0, 0], dtype=np.float64)
        elif type(position) is not np.ndarray:
            self.position = np.array(object=position, dtype=float)
        elif not position.dtype == np.float64:
            try:
                self.position = position.astype(np.float64)
            except TypeError:
                print('init_position:', position)
                print('type of init_position:', type(position))
                raise TypeError('Type of init_position of plane is wrong')
        else:
            self.position = position

        if velocity is None:
            self.velocity = np.array(object=[10, 0, np.pi / 2], dtype=np.float64)
        elif type(velocity) is not np.ndarray:
            self.velocity = np.array(object=velocity, dtype=float)
        elif not velocity.dtype == np.float64:
            try:
                self.velocity = velocity.astype(np.float64)
            except TypeError:
                print('init_velocity:', velocity)
                print('type of init_velocity:', type(velocity))
                raise TypeError('Type of init_velocity of plane is wrong')
        else:
            velocity = regularization(velocity)
            self.velocity = velocity

        if ubs is None:
            self.ubs = (10, np.pi / 12, np.pi / 24)
        else:
            self.ubs = ubs

        self.velocity_limit = velocity_limit

    # 通过输入值（速度改变量）和时间间隔计算并返回下一时刻的飞机
    def calculate_updated_plane(self, values: Union[list, tuple, np.ndarray], time: float):
        if type(values) is not np.ndarray:
            values = np.array(values)
        updated_plane = BasePlane()
        updated_plane.velocity = self.velocity + values
        updated_plane.velocity[0] = min(float(updated_plane.velocity[0]), self.velocity_limit)
        updated_plane.velocity = regularization(updated_plane.velocity)
        updated_plane.position = self.position + updated_plane.velocity * time
        return updated_plane

    def __del__(self):
        del self.position
        del self.velocity
        del self.ubs
        del self.velocity_limit
