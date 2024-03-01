import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sko.PSO import PSO
from tqdm import tqdm
from PriPy.priority import Priority

size = 2
iter_num = 10  # 迭代步数
size_pop = 100 * size  # 粒子数量
max_iter = 300  # 粒子群算法最大迭代数
w = 0.8
c1 = 0.5
c2 = 0.6  # 粒子群算法权重
c_time = 0.5  # 间隔时间

priority = Priority(size=size, time=c_time)

tra_x = np.zeros(shape=(iter_num + 1, 3 * size), dtype=np.float64)
tra_y = np.zeros(shape=(iter_num + 1, 3 * size), dtype=np.float64)  # 记录飞机轨迹

vec_x = np.zeros(shape=(iter_num + 1, 3 * size), dtype=np.float64)
vec_y = np.zeros(shape=(iter_num + 1, 3 * size), dtype=np.float64)  # 记录飞机速度


# 优化目标函数，即纳什函数
def m_nash_func(x):
    x0 = list(x[0: 3 ** size - 1])
    y0 = list(x[3 ** size - 1: 2 * 3 ** size - 2])

    x0.append(1 - sum(x0))
    y0.append(1 - sum(y0))

    if x0[3 ** size - 1] < 0 or y0[3 ** size - 1] < 0:
        return 1000

    x_t = np.array(object=x0, dtype=float)
    y_t = np.array(object=y0, dtype=float)

    return (max(m_matrix_function(x=x_t, y=y_t, matrix=priority.matrix_x), float(0)) +
            max(m_matrix_function(x=y_t, y=x_t, matrix=priority.matrix_y), float(0)))


def m_nash_func2(x):
    x0 = list(x[0: 3 ** size])
    y0 = list(x[3 ** size: 2 * 3 ** size])

    x_t = np.array(object=x0, dtype=float)
    y_t = np.array(object=y0, dtype=float)

    x_t = (x_t - np.mean(x_t)) / np.std(x_t)
    y_t = (y_t - np.mean(y_t)) / np.std(y_t)

    return (max(m_matrix_function(x=x_t, y=y_t, matrix=priority.matrix_x), float(0)) +
            max(m_matrix_function(x=y_t, y=x_t, matrix=priority.matrix_y), float(0)))


# 优化函数中涉及的两个子函数
def m_matrix_function(x: np.ndarray, y: np.ndarray, matrix: np.ndarray):
    temp = list(map(lambda i: (matrix[i] @ y.T) - (x @ matrix @ y.T), range(3 ** size)))

    return max(temp)


# 通过概率随机选择方法
def choose(x: np.ndarray, num: np.ndarray):
    result = np.array([[0, 0]])
    if x[0, 0] > num[0]:
        result[0, 0] = 0
    if x[1, 0] > num[1]:
        result[0, 1] = 0

    for i in range(1, 3 ** size):
        if sum(x[0, 0:i + 1]) > num[0] >= sum(x[0, 0:i]):
            result[0, 0] = i
        if sum(x[1, 0:i + 1]) > num[1] >= sum(x[1, 0:i]):
            result[0, 1] = i
    return result


# 选择概率最大的方法
def another_choose(x: np.ndarray):
    result = np.array(object=[0, 0], dtype=int)
    result[0] = np.where(x[0, :] == np.max(x[0, :]))[0][0]
    result[1] = np.where(x[1, :] == np.max(x[1, :]))[0][0]
    return result


def main(m_iter: int = 50):
    for i in range(size):
        tra_x[0, 3 * i: 3 * i + 3] = priority.planes_x[i].position
        tra_y[0, 3 * i: 3 * i + 3] = priority.planes_y[i].position

        vec_x[0, 3 * i: 3 * i + 3] = priority.planes_x[i].velocity
        vec_y[0, 3 * i: 3 * i + 3] = priority.planes_y[i].velocity

    pbar = tqdm(range(m_iter), desc='Processing main', ncols=100)  # 进度条

    for steps in pbar:
        time1 = time.time()
        priority.calculate_matrix()
        time4 = time.time()
        print('values calculation time:' + str(time4 - time1))

        pso = PSO(func=m_nash_func2, n_dim=2 * 3 ** size, pop=size_pop, max_iter=max_iter,
                  lb=[0] * (2 * 3 ** size), ub=[1] * (2 * 3 ** size),
                  w=w, c1=c1, c2=c2)
        pso.run()
        time2 = time.time()
        print('main pso time:' + str(time2 - time4))

        # 若优化后的最小值仍然碰到罚函数则再进行优化
        while pso.gbest_y > 10:
            pso = PSO(func=m_nash_func2, n_dim=2 * 3 ** size, pop=size_pop, max_iter=max_iter,
                      lb=[0] * (2 * 3 ** size), ub=[1] * (2 * 3 ** size),
                      w=w, c1=c1, c2=c2)
            pso.run()

        time3 = time.time()
        print('more pso time:' + str(time3 - time2))

        best_x = list(pso.gbest_x)
        possibility = np.array(object=[best_x[0: 3 ** size], best_x[3 ** size: 2 * 3 ** size]], dtype=float)

        # choice_number = np.random.rand(2)
        # choice = choose(best_x, choice_number)

        choice = another_choose(possibility)

        for i in range(size):
            priority.planes_x[i] = priority.planes_x[i].calculate_updated_plane(
                [(priority.planes_x[i].values[j] if j == (choice[0] % (3 ** i)) else 0) for j in range(3)],
                time=c_time)
            priority.planes_y[i] = priority.planes_y[i].calculate_updated_plane(
                [(priority.planes_y[i].values[j] if j == (choice[1] % (3 ** i)) else 0) for j in range(3)],
                time=c_time)
        # 更新位置和速度

        for i in range(size):
            tra_x[steps + 1, 3 * i: 3 * i + 3] = priority.planes_x[i].position
            tra_y[steps + 1, 3 * i: 3 * i + 3] = priority.planes_y[i].position

            vec_x[steps + 1, 3 * i: 3 * i + 3] = priority.planes_x[i].velocity
            vec_y[steps + 1, 3 * i: 3 * i + 3] = priority.planes_y[i].velocity  # 记录


# 画图
def m_draw():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    pbar = tqdm(range(iter_num), desc='Processing draw', ncols=100)
    for i in pbar:
        for j in range(size):
            ax.plot(tra_x[max(0, i - 4): i + 1, 0 + 3 * j],
                    tra_x[max(0, i - 4): i + 1, 1 + 3 * j],
                    tra_x[max(0, i - 4): i + 1, 2 + 3 * j],
                    c='r')
            ax.scatter(tra_x[i, 0 + 3 * j],
                       tra_x[i, 1 + 3 * j],
                       tra_x[i, 2 + 3 * j],
                       c='r', depthshade=True, marker="^", s=30)
            ax.plot(tra_y[max(0, i - 4): i + 1, 0 + 3 * j],
                    tra_y[max(0, i - 4): i + 1, 1 + 3 * j],
                    tra_y[max(0, i - 4): i + 1, 2 + 3 * j],
                    c='b')
            ax.scatter(tra_y[i, 0 + 3 * j],
                       tra_y[i, 1 + 3 * j],
                       tra_y[i, 2 + 3 * j],
                       c='b', depthshade=True, marker="^", s=30)
        name = './Outputs/Pictures/trac' + str(i + 1) + '.png'
        plt.savefig(name)
        plt.cla()
    # plt.show()


def m_draw_fig_based_excel():
    print('data reading')
    sheet_names = ['trac_x', 'trac_y', 'vec_x', 'vec_y']
    datas = [pd.read_excel('./Outputs/data.xlsx', sheet_name=i).values[:, 1::] for i in sheet_names]
    # print(datas)
    # print(datas[0].shape)
    interval = 50
    frames = int(c_time * iter_num * 1000 / interval)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # x_ticks = np.linspace(start=max(max([max(datas[0][:, 3 * i]) for i in range(size)]),
    #                                 max([max(datas[1][:, 3 * i]) for i in range(size)])),
    #                       stop=min(min([min(datas[0][:, 3 * i]) for i in range(size)]),
    #                                min([min(datas[1][:, 3 * i]) for i in range(size)])),
    #                       num=10)
    # y_ticks = np.linspace(start=max(max([max(datas[0][:, 3 * i + 1]) for i in range(size)]),
    #                                 max([max(datas[1][:, 3 * i + 1]) for i in range(size)])),
    #                       stop=min(min([min(datas[0][:, 3 * i + 1]) for i in range(size)]),
    #                                min([min(datas[1][:, 3 * i + 1]) for i in range(size)])),
    #                       num=10)
    # z_ticks = np.linspace(start=max(max([max(datas[0][:, 3 * i + 2]) for i in range(size)]),
    #                                 max([max(datas[1][:, 3 * i + 2]) for i in range(size)])),
    #                       stop=min(min([min(datas[0][:, 3 * i + 2]) for i in range(size)]),
    #                                min([min(datas[1][:, 3 * i + 2]) for i in range(size)])),
    #                       num=10)
    points_x = [list(datas[0][0, :])]
    points_y = [list(datas[1][0, :])]
    draws = [(plt.scatter([points_x[0][3 * i] for i in range(size)],
                          [points_x[0][1 + 3 * i] for i in range(size)],
                          [points_x[0][2 + 3 * i] for i in range(size)],
                          c='r', depthshade=True, marker="^")
              ), ax.scatter([points_y[0][3 * i] for i in range(size)],
                            [points_y[0][1 + 3 * i] for i in
                             range(size)],
                            [points_y[0][2 + 3 * i] for i in
                             range(size)],
                            c='b', depthshade=True, marker="^")]
    for i in range(size):
        draws.append(ax.plot(points_x[0][3 * i],
                             points_x[0][1 + 3 * i],
                             points_x[0][2 + 3 * i],
                             c='r')
                     )

        draws.append(ax.plot(points_y[0][3 * i],
                             points_y[0][1 + 3 * i],
                             points_y[0][2 + 3 * i],
                             c='b')
                     )

    def init_fun():
        return tuple(draws)

    def update(num):
        if num == 0:
            return init_fun()
        else:
            base_time = int((num * interval) // (c_time * 1000))
            # more_time = int(num - base_time)
            a_x = datas[2][base_time + 1] - datas[2][base_time]
            a_y = datas[3][base_time + 1] - datas[3][base_time]
            num = int(num)
            # print(num)
            # print(np.array(points_x[num - 1]) +
            #                      datas[2][base_time] * (more_time / 1000) + 0.5 * a_x * (more_time / 1000) ** 2)
            # print(list(np.array(points_x[num - 1]) +
            #                      datas[2][base_time] * (more_time / 1000) + 0.5 * a_x * (more_time / 1000) ** 2))

            points_x.append(list(np.array(points_x[num - 1]) +
                                 datas[2][base_time] * float(interval / 1000) + 0.5 * a_x * float(
                interval / 1000) ** 2))
            points_y.append(list(np.array(points_y[num - 1]) +
                                 datas[3][base_time] * float(interval / 1000) + 0.5 * a_y * float(
                interval / 1000) ** 2))
            draws[0] = ax.scatter([points_x[num][3 * i] for i in range(size)],
                                  [points_x[num][1 + 3 * i] for i in range(size)],
                                  [points_x[num][2 + 3 * i] for i in range(size)],
                                  c='r', depthshade=True, marker="^", s=30)
            draws[1] = ax.scatter([points_y[num][3 * i] for i in range(size)],
                                  [points_y[num][1 + 3 * i] for i in range(size)],
                                  [points_y[num][2 + 3 * i] for i in range(size)],
                                  c='b', depthshade=True, marker="^", s=30)
            for i in range(size):
                draws[2 + 2 * i] = ax.plot([points_x[j][3 * i] for j in range(num)],
                                           [points_x[j][1 + 3 * i] for j in range(num)],
                                           [points_x[j][2 + 3 * i] for j in range(num)],
                                           c='r')
                draws[3 + 2 * i] = ax.plot([points_y[j][3 * i] for j in range(num)],
                                           [points_y[j][1 + 3 * i] for j in range(num)],
                                           [points_y[j][2 + 3 * i] for j in range(num)],
                                           c='b'
                                           )
            return tuple(draws)

    for k in range(frames):
        update(k)
        name = './Outputs/Fig_Res/Pictures/a' + str(k) + '.png'
        plt.savefig(name)
        plt.cla()
    with imageio.get_writer(uri='./Outputs/test.gif', mode='I', fps=20) as writer:
        pbar = tqdm(range(frames), desc='Processing gif draw', ncols=100)
        for i in pbar:
            writer.append_data(imageio.v3.imread('./Outputs/Fig_Res/Pictures/a' + str(i) + '.png'))
    # ani = FuncAnimation(fig
    #                     , update
    #                     , init_func=init_fun
    #                     , frames=np.linspace(1, frames, frames)
    #                     , interval=interval
    #                     , blit=True
    #                     )
    # ani.save("4.gif", fps=500, writer="imagemagick")

    # plt.show()
    # ani.save("animation.mp4", fps=20, writer="ffmpeg")


# 保存数据到excel
def data_save():
    print('data saving')
    writer = pd.ExcelWriter('./Outputs/data.xlsx')
    data1 = pd.DataFrame(tra_x)
    data1.to_excel(writer, sheet_name='trac_x', float_format='%.5f', )

    data2 = pd.DataFrame(tra_y)
    data2.to_excel(writer, sheet_name='trac_y', float_format='%.5f')
    data3 = pd.DataFrame(vec_x)
    data3.to_excel(writer, sheet_name='vec_x', float_format='%.5f')
    data4 = pd.DataFrame(vec_y)
    data4.to_excel(writer, sheet_name='vec_y', float_format='%.5f')

    writer.close()


if __name__ == '__main__':
    total_s_time = time.time()  # 计算总用时
    main(iter_num)
    total_e_time = time.time()
    print('耗时:{}s'.format(total_e_time - total_s_time))
    m_draw()
    data_save()
    m_draw_fig_based_excel()
