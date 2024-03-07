import numpy as np
from matplotlib import pyplot as plt
from sko.PSO import PSO

from pripy3.base_calculate_func import BaseCalculateFunc
from pripy3.plane import Plane
from pripy3.util import single_priority, BasePlane
from pripy3.PriPy3 import PriPy3

size = 4  # 飞机群规模
iter_num = 200  # 迭代步数
pop_size = 200  # 计算纳什函数的pso粒子数量
max_iter = 50  # 计算纳什函数pso的最大迭代数
w = 0.8  # 计算纳什函数pso的w
c1 = 0.5  # 计算纳什函数pso的c1
c2 = 0.6  # 计算纳什函数pso的c2
time_interval = 0.1  # 间隔时间


# 下面这个函数是对飞机计算最佳数值的覆写，这个方法必须包含在BaseCalculateFunc的子类中，下面先定义了其一个子类PsoCalculateFunc
class PsoCalculateFunc(BaseCalculateFunc):
    @staticmethod  # 该声明，@staticmethod，是必须的
    # 下面需要override父类函数def calculate_func(plane_x: BasePlane, planes_y: list, k: list = None, **kwargs) -> float:
    # 不能改变参数列表，额外的参数都需要从**kwargs传进
    # 该函数和所需参数需要在生成飞机时传入
    # 前三个参数不需要手动传入
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


# 生成飞机群x，参数有：
# position（位置）、velocity（速度）、velocity_limit（最大速度）、ubs（飞机性能，这里三个参数分别为就是速度三个参数的最大改变量）
# 若要改变飞机计算最佳数值的方法请参考example_with_calculate.py
# 生成飞机时传入计算的函数和参数
m_planes_x = [Plane(position=(-10 - 8 * i, 0, 0),
                    velocity=(5 - 0.6 * i, 0, np.pi / 2),
                    velocity_limit=10, ubs=(0.1, np.pi / 24, np.pi / 24),
                    func=PsoCalculateFunc.calculate_func,
                    k=[0.1, 0.3, 0.6],
                    pop_size=50, max_iter=100, w=0.8, c1=0.5, c2=0.5
                    ) for i in range(size)]

# 生成飞机群y
m_planes_y = [Plane(position=(7 + 5 * i, 0.2 * i, 5 + 0.5 * i),
                    velocity=(5 - 0.6 * i, np.pi, np.pi / 2),
                    velocity_limit=10, ubs=(0.1, np.pi / 24, np.pi / 24),
                    func=PsoCalculateFunc.calculate_func,
                    k=[0.1, 0.3, 0.6],
                    pop_size=50, max_iter=100, w=0.8, c1=0.5, c2=0.5
                    ) for i in range(size)]

# 接下来开始迭代，PriPy3是一个迭代器，类似于range()，将飞机群等上面的所有参数传进去生成迭代器。
# 每一次for循环都会进行一次迭代，返回的i为一个自定义类，里面包含每一次迭代的飞机数据，在for循环内部就可以进行一些自定的操作
# 在每一次迭代都会自动画图，画出最近4次的轨迹，保存在./outputs/pictures;画出动图,保存在./outputs/gif_res;
# 最后的数据会保存在./outputs/data.xlsx中
# 下面的例子在迭代的过程中，输出迭代步数，并在最后一次迭代时画出总体轨迹并展示出来（不保存，仅用于观察）
# 注：在对该图进行操作后（如关闭，保存）后程序才会继续运行
for i in PriPy3(planes_x=m_planes_x, planes_y=m_planes_y, size=size,
                k_x=[0.55, 0.05, 0.4], k_y=[0.05, 0.25, 0.7],
                iter_num=iter_num, time_interval=time_interval,
                pso_pop_size=pop_size, pso_max_iter=max_iter, pso_w=w, pso_c1=c1, pso_c2=c2):
    # steps是i的属性，是当前的迭代步数
    print(i.steps)

    # iter_num是i的属性，是最大的迭代步数，但是由于i从0开始，所以到max_iter - 1 就是最后一次迭代，如同range()。
    # trac_x, trac_y, vec_x, vec_y分别是x的位置，y的位置，x的速度，y的速度
    if i.steps == i.iter_num - 1:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for plane_num in range(size):
            ax.plot(i.trac_x[:, 0 + 3 * plane_num],
                    i.trac_x[:, 1 + 3 * plane_num],
                    i.trac_x[:, 2 + 3 * plane_num],
                    c='r')
            ax.scatter(i.trac_x[i.iter_num, 0 + 3 * plane_num],
                       i.trac_x[i.iter_num, 1 + 3 * plane_num],
                       i.trac_x[i.iter_num, 2 + 3 * plane_num],
                       c='r', depthshade=True, marker="^", s=30)
            ax.plot(i.trac_y[:, 0 + 3 * plane_num],
                    i.trac_y[:, 1 + 3 * plane_num],
                    i.trac_y[:, 2 + 3 * plane_num],
                    c='b')
            ax.scatter(i.trac_y[i.iter_num, 0 + 3 * plane_num],
                       i.trac_y[i.iter_num, 1 + 3 * plane_num],
                       i.trac_y[i.iter_num, 2 + 3 * plane_num],
                       c='b', depthshade=True, marker="^", s=30)
        plt.show()
