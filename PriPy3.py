import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sko.PSO import PSO
from tqdm import tqdm

from PlanePy.util import translate_velocity
from PriPy.priority import Priority
from PriPy.priority import base3_transform


class PriPy3:
    def __init__(self, planes_x=None, planes_y=None, k_x=None, k_y=None,
                 size=2, iter_num=10, time_interval=0.1,
                 pso_pop_size=200, pso_max_iter=100, pso_w=0.5, pso_c1=0.6, pso_c2=0.6):

        self.priority = Priority(planes_x=planes_x, planes_y=planes_y, size=size,
                                 k_x=k_x, k_y=k_y, time=time_interval)

        (self.iter_num, self.size, self.time_interval,
         self.pso_pop_size, self.pso_max_iter, self.pso_w, self.pso_c1, self.pso_c2) = \
            iter_num, size, time_interval, pso_pop_size, pso_max_iter, pso_w, pso_c1, pso_c2

        self.trac_x, self.trac_y, self.vec_x, self.vec_y = \
            (np.zeros(shape=(self.iter_num + 1, 3 * size), dtype=np.float64),
             np.zeros(shape=(self.iter_num + 1, 3 * size), dtype=np.float64),
             np.zeros(shape=(self.iter_num + 1, 3 * size), dtype=np.float64),
             np.zeros(shape=(self.iter_num + 1, 3 * size), dtype=np.float64))

        for i in range(size):
            self.trac_x[0, 3 * i: 3 * i + 3] = self.priority.planes_x[i].position
            self.trac_y[0, 3 * i: 3 * i + 3] = self.priority.planes_y[i].position

            self.vec_x[0, 3 * i: 3 * i + 3] = self.priority.planes_x[i].velocity
            self.vec_y[0, 3 * i: 3 * i + 3] = self.priority.planes_y[i].velocity

    def __iter__(self):
        self.steps = -1
        dirs = ('./outputs', './outputs/pictures', './outputs/gif_res', './outputs/gif_res/pictures')
        for dir_name in dirs:
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
        return self

    def __next__(self):
        self.steps += 1
        if self.steps < self.iter_num:
            self.priority.calculate_matrix()
            pso = PSO(func=self.m_nash_func, n_dim=2 * 3 ** self.size, pop=self.pso_pop_size,
                      max_iter=self.pso_max_iter,
                      lb=[0] * (2 * 3 ** self.size), ub=[1] * (2 * 3 ** self.size),
                      w=self.pso_w, c1=self.pso_c1, c2=self.pso_c2)
            pso.run()
            while pso.gbest_y > 10:
                pso = PSO(func=self.m_nash_func, n_dim=2 * 3 ** self.size, pop=self.pso_pop_size,
                          max_iter=self.pso_max_iter,
                          lb=[0] * (2 * 3 ** self.size), ub=[1] * (2 * 3 ** self.size),
                          w=self.pso_w, c1=self.pso_c1, c2=self.pso_c2)
                pso.run()

            best_x = list(pso.gbest_x)
            possibility = np.array(object=[best_x[0: 3 ** self.size], best_x[3 ** self.size: 2 * 3 ** self.size]],
                                   dtype=float)
            possibility[0, :] = possibility[0, :] / np.sum(possibility[0, :])
            possibility[1, :] = possibility[1, :] / np.sum(possibility[1, :])

            choice_number = np.random.rand(2)
            choice = self.choose(possibility, choice_number)
            result_x = base3_transform(choice[0], size=self.size)
            result_y = base3_transform(choice[1], size=self.size)

            for i in range(self.size):
                self.priority.planes_x[i] = self.priority.planes_x[i].calculate_updated_plane(
                        [(self.priority.planes_x[i].values[j] if j == (result_x[i]) else 0) for j in range(3)],
                        time=self.time_interval)
                self.priority.planes_y[i] = self.priority.planes_y[i].calculate_updated_plane(
                        [(self.priority.planes_y[i].values[j] if j == (result_y[i]) else 0) for j in range(3)],
                        time=self.time_interval)

            for i in range(self.size):
                self.trac_x[self.steps + 1, 3 * i: 3 * i + 3] = self.priority.planes_x[i].position
                self.trac_y[self.steps + 1, 3 * i: 3 * i + 3] = self.priority.planes_y[i].position

                self.vec_x[self.steps + 1, 3 * i: 3 * i + 3] = self.priority.planes_x[i].velocity
                self.vec_y[self.steps + 1, 3 * i: 3 * i + 3] = self.priority.planes_y[i].velocity

            self.m_draw()
            if self.steps == self.iter_num - 1:
                self.data_save()
                self.m_draw_gif()
            return self
        else:
            raise StopIteration()

    def m_matrix_function(self, x: np.ndarray, y: np.ndarray, matrix: np.ndarray):
        temp = list(map(lambda i: (matrix[i] @ y.T) - (x @ matrix @ y.T), range(3 ** self.size)))

        return max(temp)

    def m_nash_func(self, x):
        x0 = list(x[0: 3 ** self.size])
        y0 = list(x[3 ** self.size: 2 * 3 ** self.size])

        x_t = np.array(object=x0, dtype=float)
        y_t = np.array(object=y0, dtype=float)

        x_t = x_t / np.sum(x_t)
        y_t = y_t / np.sum(y_t)

        return (max(self.m_matrix_function(x=x_t, y=y_t, matrix=self.priority.matrix_x), float(0)) +
                max(self.m_matrix_function(x=y_t, y=x_t, matrix=self.priority.matrix_y), float(0)))

    def choose(self, x: np.ndarray, num: np.ndarray):
        result = np.array([0, 0])
        if x[0, 0] > num[0]:
            result[0] = 0
        if x[1, 0] > num[1]:
            result[1] = 0

        for i in range(1, 3 ** self.size):
            if sum(x[0, 0:i + 1]) > num[0] >= sum(x[0, 0:i]):
                result[0] = i
            if sum(x[1, 0:i + 1]) > num[1] >= sum(x[1, 0:i]):
                result[1] = i
        return result

    def m_draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=18.0, azim=-160.0)
        for i in range(self.size):
            ax.plot(self.trac_x[max(0, self.steps - 4): self.steps + 1, 0 + 3 * i],
                    self.trac_x[max(0, self.steps - 4): self.steps + 1, 1 + 3 * i],
                    self.trac_x[max(0, self.steps - 4): self.steps + 1, 2 + 3 * i],
                    c='r')
            ax.scatter(self.trac_x[self.steps, 0 + 3 * i],
                       self.trac_x[self.steps, 1 + 3 * i],
                       self.trac_x[self.steps, 2 + 3 * i],
                       c='r', depthshade=True, marker="^", s=30)
            ax.plot(self.trac_y[max(0, self.steps - 4): self.steps + 1, 0 + 3 * i],
                    self.trac_y[max(0, self.steps - 4): self.steps + 1, 1 + 3 * i],
                    self.trac_y[max(0, self.steps - 4): self.steps + 1, 2 + 3 * i],
                    c='b')
            ax.scatter(self.trac_y[self.steps, 0 + 3 * i],
                       self.trac_y[self.steps, 1 + 3 * i],
                       self.trac_y[self.steps, 2 + 3 * i],
                       c='b', depthshade=True, marker="^", s=30)
        # if self.steps == self.iter_num - 1:
        #     plt.show()
        name = './outputs/pictures/trajectory' + str(self.steps + 1) + '.png'
        plt.savefig(name)
        plt.close()

    def data_save(self):
        print('data saving')
        writer = pd.ExcelWriter('./outputs/data.xlsx')
        data1 = pd.DataFrame(self.trac_x)
        data1.to_excel(writer, sheet_name='trac_x', float_format='%.5f', )

        data2 = pd.DataFrame(self.trac_y)
        data2.to_excel(writer, sheet_name='trac_y', float_format='%.5f')
        data3 = pd.DataFrame(self.vec_x)
        data3.to_excel(writer, sheet_name='vec_x', float_format='%.5f')
        data4 = pd.DataFrame(self.vec_y)
        data4.to_excel(writer, sheet_name='vec_y', float_format='%.5f')

        writer.close()

    def m_draw_gif(self):
        fps = 20
        interval = int(1000 / fps)
        frames = int(self.time_interval * self.iter_num * 1000 / interval)
        points_x = np.zeros(shape=(frames, 3 * self.size))
        points_y = np.zeros(shape=(frames, 3 * self.size))
        points_x[0, :] = self.trac_x[0, :]
        points_y[0, :] = self.trac_y[0, :]

        def multi_transform_velocity(multi_velocity):
            new_velocity = multi_velocity.copy()
            for num_planes in range(self.size):
                print(num_planes)
                print(type(num_planes))
                new_velocity[3 * num_planes: 3 * num_planes + 3] = translate_velocity(
                    new_velocity[3 * num_planes: 3 * num_planes + 3])
            return new_velocity

        for num in range(1, frames):
            base_time = int((num * interval) // (self.time_interval * 1000))
            if num * interval == base_time * self.time_interval * 1000:
                points_x[num] = self.trac_x[base_time]
                points_y[num] = self.trac_y[base_time]
            else:
                ratio = (float(num * interval / 1000) - base_time * self.time_interval) / float(self.time_interval)
                print(ratio)
                v_x = ((1 - ratio) * multi_transform_velocity(self.vec_x[base_time, :]) +
                       ratio * multi_transform_velocity(self.vec_x[base_time + 1, :]))
                v_y = ((1 - ratio) * multi_transform_velocity(self.vec_y[base_time, :]) +
                       ratio * multi_transform_velocity(self.vec_y[base_time + 1, :]))

                points_x[num] = (self.trac_x[base_time, :] +
                                 0.5 * (float(num * interval / 1000) - base_time * self.time_interval) *
                                 (self.vec_x[base_time, :] + v_x))
                points_y[num] = (self.trac_y[base_time, :] +
                                 0.5 * (float(num * interval / 1000) - base_time * self.time_interval) *
                                 (self.vec_y[base_time, :] + v_y))

        print(self.trac_x)
        print(points_x)


        def frame_draw(frame_num):
            if frame_num == 0:
                for draw_plane_index in range(self.size):
                    ax.scatter(points_x[frame_num, 3 * draw_plane_index],
                               points_x[frame_num, 1 + 3 * draw_plane_index],
                               points_x[frame_num, 2 + 3 * draw_plane_index],
                               c='r', depthshade=True, marker="^", s=30)
                    ax.scatter(points_y[frame_num, 3 * draw_plane_index],
                               points_y[frame_num, 1 + 3 * draw_plane_index],
                               points_y[frame_num, 2 + 3 * draw_plane_index],
                               c='b', depthshade=True, marker="^", s=30)
            else:
                for draw_plane_index in range(self.size):
                    temp_lb = max(0, frame_num - int(4 * self.time_interval * 1000 / interval))
                    ax.plot(
                        points_x[temp_lb: frame_num + 1, 3 * draw_plane_index],
                        points_x[temp_lb: frame_num + 1, 1 + 3 * draw_plane_index],
                        points_x[temp_lb: frame_num + 1, 2 + 3 * draw_plane_index],
                        c='r')
                    ax.scatter(points_x[frame_num, 3 * draw_plane_index],
                               points_x[frame_num, 1 + 3 * draw_plane_index],
                               points_x[frame_num, 2 + 3 * draw_plane_index],
                               c='r', depthshade=True, marker="^", s=30)
                    ax.plot(
                        points_y[temp_lb: frame_num + 1, 3 * draw_plane_index],
                        points_y[temp_lb: frame_num + 1, 1 + 3 * draw_plane_index],
                        points_y[temp_lb: frame_num + 1, 2 + 3 * draw_plane_index],
                        c='b')
                    ax.scatter(points_y[frame_num, 3 * draw_plane_index],
                               points_y[frame_num, 1 + 3 * draw_plane_index],
                               points_y[frame_num, 2 + 3 * draw_plane_index],
                               c='b', depthshade=True, marker="^", s=30)

        for k in range(frames):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=18.0, azim=-160.0)
            frame_draw(k)
            name = './outputs/gif_res/pictures/a' + str(k) + '.png'
            plt.savefig(name)
            plt.close()

        with imageio.get_writer(uri='./outputs/gif_res/trajectory.gif', mode='I', fps=fps) as writer:
            pbar = tqdm(range(frames), desc='Processing gif draw', ncols=100)
            for i in pbar:
                writer.append_data(imageio.v3.imread('./outputs/gif_res/pictures/a' + str(i) + '.png'))


if __name__ == '__main__':
    for m_p in PriPy3(iter_num=100):
        print(m_p.steps)
