import numpy as np

from pripy3.plane import Plane
from pripy3.util import single_priority, base3_transform


class Priority:
    def __init__(self,
                 planes_x: list[Plane] = None,
                 planes_y: list[Plane] = None,
                 size: int = 1,
                 k_x: list = None, k_y: list = None,
                 time: float = 0.1):
        if planes_x is None:
            planes_x = (Plane(position=[-10 - 8 * i, 0, 0],
                              velocity=[5 - 0.6 * i, 0, np.pi / 2]) for i in range(size))
        if planes_y is None:
            planes_y = (Plane(position=[7 + 5 * i, 0.2 * i, 5 + 0.5 * i],
                              velocity=[5 - 0.6 * i, np.pi, np.pi / 2]) for i in range(size))
        self.planes_x = planes_x
        self.planes_y = planes_y
        self.size = size
        self.__matrix_x = np.zeros(shape=(3 ** size, 3 ** size), dtype=np.float64)
        self.__matrix_y = np.zeros(shape=(3 ** size, 3 ** size), dtype=np.float64)
        self.k_x = k_x or [0.55, 0.05, 0.4]
        self.k_y = k_y or [0.05, 0.25, 0.7]
        self.time = time

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

        for i in range(3 ** self.size):
            result_x = base3_transform(i, size=self.size)
            updated_planes_x = [
                self.planes_x[m].base_plane.calculate_updated_plane(self.planes_x[m].values[result_x[m]],
                                                                    time=self.time) for m in range(self.size)]

            for j in range(3 ** self.size):
                result_y = base3_transform(j, size=self.size)
                updated_planes_y = [
                    self.planes_y[n].base_plane.calculate_updated_plane(self.planes_x[n].values[result_y[n]],
                                                                        time=self.time) for n in range(self.size)]

                self.__matrix_x[i, j] = np.mean(
                    [single_priority(updated_planes_x[p], updated_planes_y[q], self.k_x) for p in range(self.size) for q
                     in range(self.size)])
                self.__matrix_y[i, j] = np.mean(
                    [single_priority(updated_planes_y[p], updated_planes_x[q], self.k_y) for p in range(self.size) for q
                     in range(self.size)])

        return self.__matrix_x, self.__matrix_y
