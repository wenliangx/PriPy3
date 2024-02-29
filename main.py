import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
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
c_time = 1

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
            max(m_matrix_function(x=y_t, y=x_t, matrix=priority.matrix_x), float(0)))


def m_nash_func2(x):
    x0 = list(x[0: 3 ** size])
    y0 = list(x[3 ** size: 2 * 3 ** size])

    x_t = np.array(object=x0, dtype=float)
    y_t = np.array(object=y0, dtype=float)

    x_t = (x_t - np.mean(x_t)) / np.std(x_t)
    y_t = (y_t - np.mean(y_t)) / np.std(y_t)

    return (max(m_matrix_function(x=x_t, y=y_t, matrix=priority.matrix_x), float(0)) +
            max(m_matrix_function(x=y_t, y=x_t, matrix=priority.matrix_x), float(0)))


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
        print('run time:' + str(time4 - time1))

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
    ax = Axes3D(fig)
    fig.add_axes(ax)

    pbar = tqdm(range(iter_num), desc='Processing draw', ncols=100)
    for i in pbar:
        for j in range(size):
            ax.plot(tra_x[max(0, i - 4): i + 1, 0 + 3 * j],
                    tra_x[max(0, i - 4): i + 1, 1 + 3 * j],
                    tra_x[max(0, i - 4): i + 1, 2 + 3 * j],
                    c='r')
            ax.plot(tra_y[max(0, i - 4): i + 1, 0 + 3 * j],
                    tra_y[max(0, i - 4): i + 1, 1 + 3 * j],
                    tra_y[max(0, i - 4): i + 1, 2 + 3 * j],
                    c='b')
        name = './Outputs/Pictures/trac' + str(i + 1) + '.png'
        plt.savefig(name)
        plt.cla()
    # plt.show()


# 保存数据到excel
def data_save():
    writer = pd.ExcelWriter('./Outputs/data.xlsx')
    data1 = pd.DataFrame(tra_x)
    data1.to_excel(writer, sheet_name='trac_x', float_format='%.5f')

    data2 = pd.DataFrame(tra_y)
    data2.to_excel(writer, sheet_name='trac_y', float_format='%.5f')
    data3 = pd.DataFrame(vec_x)
    data3.to_excel(writer, sheet_name='vec_x', float_format='%.5f')
    data4 = pd.DataFrame(vec_x)
    data4.to_excel(writer, sheet_name='vec_y', float_format='%.5f')

    writer.close()


if __name__ == '__main__':
    total_s_time = time.time()  # 计算总用时
    main(iter_num)
    total_e_time = time.time()
    print('耗时:{}s'.format(total_e_time - total_s_time))
    m_draw()
    data_save()
