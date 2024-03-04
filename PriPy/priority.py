import numpy as np

from PlanePy.plane import Plane
from PriPy.util import single_priority


class Priority:
    def __init__(self, planes_x: list[Plane] = None, planes_y: list[Plane] = None,
                 size: int = 1,
                 k_x: list = None, k_y: list = None,
                 time: float = 0.1):
        if planes_x is None:
            planes_x = list(map(
                lambda i:
                Plane(position=[-10 - 8 * i, 0, 0],
                      velocity=[5 - 0.6 * i, 0, np.pi / 2]),
                range(size)))
        if planes_y is None:
            planes_y = list(map(
                lambda i:
                Plane(position=[7 + 5 * i, 0.2 * i, 5 + 0.5 * i],
                      velocity=[5 - 0.6 * i, np.pi, np.pi / 2]),
                range(size)))
        self.planes_x = planes_x
        self.planes_y = planes_y
        self.__matrix_x = np.zeros(shape=(3 ** size, 3 ** size), dtype=np.float64)
        self.__matrix_y = np.zeros(shape=(3 ** size, 3 ** size), dtype=np.float64)

        if k_x is None:
            self.k_x = [0.55, 0.05, 0.4]
        else:
            self.k_x = k_x

        if k_y is None:
            self.k_y = [0.05, 0.25, 0.7]
        else:
            self.k_y = k_y
        self.time = time
        self.size = size

    @property
    def matrix_x(self):
        return self.__matrix_x

    @property
    def matrix_y(self):
        return self.__matrix_y

    @property
    def matrix(self):
        return self.__matrix_x, self.__matrix_y

    def calculate_matrix(self):
        for i in range(self.size):
            self.planes_x[i].calculate_values(planes_y=self.planes_y, time=self.time)
            self.planes_y[i].calculate_values(planes_y=self.planes_x, time=self.time)
        # 错误写法 这里用map会被优化导致错误
        # map(lambda m: self.planes_x[m].calculate_values(planes_y=self.planes_y, k=self.k_x, time=self.time),
        #     range(self.size))
        # map(lambda n: self.planes_y[n].calculate_values(planes_y=self.planes_x, k=self.k_y, time=self.time),
        #     range(self.size))
        for i in range(3 ** self.size):
            result_x = base3_transform(i, size=self.size)
            updated_planes_x = list(map(
                lambda m: self.planes_x[m].base_plane.calculate_updated_plane
                (self.planes_x[m].values[result_x[m]], time=self.time),
                range(self.size)))
            for j in range(3 ** self.size):
                result_y = base3_transform(j, size=self.size)
                updated_planes_y = list(map(
                    lambda n: self.planes_y[n].base_plane.calculate_updated_plane
                    (self.planes_x[n].values[result_y[n]], time=self.time),
                    range(self.size)))
                self.__matrix_x[i, j] = np.mean(list(map(
                    lambda p, q: single_priority(updated_planes_x[p], updated_planes_y[q], self.k_x),
                    range(self.size), range(self.size))))
                self.__matrix_y[i, j] = np.mean(list(map(
                    lambda p, q: single_priority(updated_planes_y[p], updated_planes_x[q], self.k_y),
                    range(self.size), range(self.size))))

        return self.__matrix_x, self.__matrix_y


def base3_transform(num, size):
    if size == 1:
        return [num]
    else:
        result = [int(num / (3 ** (size - 1)))] * size
        num = num - result[0] * (3 ** (size - 1))
        for i in range(1, size):
            result[i] = int(num / (3 ** (size - 1 - i)))
            num = num - result[i] * (3 ** (size - 1 - i))
        return result

