import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle
import random
class Vector2d:
    """
    2维向量, 支持加减, 支持常量乘法(右乘)
    """

    def __init__(self, x, y):
        self.deltaX = x
        self.deltaY = y
        self.length = -1
        self.direction = [0, 0]  # 方向向量
        self.vector2d_share()

    def vector2d_share(self):  # 计算方向向量
        if type(self.deltaX) == type(list()) and type(self.deltaY) == type(list()):
            deltaX, deltaY = self.deltaX, self.deltaY
            self.deltaX = deltaY[0] - deltaX[0]
            self.deltaY = deltaY[1] - deltaX[1]
            self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2) * 1.0
            if self.length > 0:
                self.direction = [self.deltaX / self.length, self.deltaY / self.length]
            else:
                self.direction = None
        else:
            self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2) * 1.0
            if self.length > 0:
                self.direction = [self.deltaX / self.length, self.deltaY / self.length]
            else:
                self.direction = None

    def __add__(self, other):  # 向量相加
        """
        + 重载
        :param other:
        :return:
        """
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX += other.deltaX
        vec.deltaY += other.deltaY
        vec.vector2d_share()
        return vec

    def __sub__(self, other):  # 向量相减
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX -= other.deltaX
        vec.deltaY -= other.deltaY
        vec.vector2d_share()
        return vec

    def __mul__(self, other):   # 向量相乘
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX *= other
        vec.deltaY *= other
        vec.vector2d_share()
        return vec

    def __truediv__(self, other):  # 向量相除
        return self.__mul__(1.0 / other)

    def __repr__(self):   # 向量输出
        return 'Vector deltaX:{}, deltaY:{}, length:{}, direction:{}'.format(self.deltaX, self.deltaY, self.length,
                                                                             self.direction)


def draw_path(Path, label='Planned Path'):
    if len(Path) < 2:
        return  # 如果路径长度小于2，不进行绘制
    # 提取 x 和 y 坐标
    x_coords = [point[0] for point in Path]
    y_coords = [point[1] for point in Path]
    # 绘制路径
    plt.plot(x_coords, y_coords, label=label, marker='o', linestyle='-', color='green')
    plt.legend()  # 显示图例


class APF:
    """
    人工势场寻路
    """

    def __init__(self, Start: (), Goal: (), obstacles: [], K_att: float, K_rep: float, r_length: float,
                 Step_size: float, Max_iters: int, Goal_threshold: float, Is_plot=False):
        """
        :param Start: 起点
        :param Goal: 终点
        :param obstacles: 障碍物列表，每个元素为Vector2d对象
        :param K_att: 引力系数
        :param K_rep: 斥力系数
        :param r_length: 斥力作用范围
        :param Step_size: 步长
        :param Max_iters: 最大迭代次数
        :param Goal_threshold: 离目标点小于此值即认为到达目标点
        :param Is_plot: 是否绘图
        """
        self.start = Vector2d(Start[0], Start[1])
        self.current_pos = Vector2d(Start[0], Start[1])  # 当前位置
        self.goal = Vector2d(Goal[0], Goal[1])
        self.obstacles = [Vector2d(Ob[0], Ob[1]) for Ob in obstacles]
        self.k_att = K_att
        self.k_rep = K_rep
        self.rr = r_length  # 斥力作用范围
        self.step_size = Step_size
        self.max_iters = Max_iters
        self.iters = 0
        self.goal_threshold = Goal_threshold
        self.path = list()
        self.is_path_plan_success = False
        self.is_plot = Is_plot
        self.delta_t = 0.01

    def attractive(self):
        """
        引力计算
        :return: 引力
        """
        att = (self.goal - self.current_pos) * self.k_att  # 方向由机器人指向目标点
        return att

    def repulsion(self):
        """
        斥力计算
        :return: 斥力大小
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for obstacle in self.obstacles:
            # obstacle = Vector2d(0, 0)
            obs_to_rob = self.current_pos - obstacle
            rob_to_goal = self.goal - self.current_pos
            if obs_to_rob.length > self.rr:  # 超出障碍物斥力影响范围
                pass
            else:
                rep_1 = Vector2d(obs_to_rob.direction[0], obs_to_rob.direction[1]) * self.k_rep * (
                        1.0 / obs_to_rob.length - 1.0 / self.rr) / (obs_to_rob.length ** 2) * (rob_to_goal.length ** 2)
                rep_2 = Vector2d(rob_to_goal.direction[0], rob_to_goal.direction[1]) * self.k_rep * (
                            (1.0 / obs_to_rob.length - 1.0 / self.rr) ** 2) * rob_to_goal.length
                rep += (rep_1 + rep_2)
        return rep

    def path_plan(self):
        """
        Path plan
        :return:
        """
        while self.iters < self.max_iters and (self.current_pos - self.goal).length > self.goal_threshold:
            f_vec = self.attractive() + self.repulsion()  # 合力
            self.current_pos += Vector2d(f_vec.direction[0], f_vec.direction[1]) * self.step_size  # 合力方向向量×步长
            self.iters += 1  # 迭代次数+1
            self.path.append([round(self.current_pos.deltaX), round(self.current_pos.deltaY)])  # 路径点加入
            if self.is_plot:
                subplot.plot(round(self.current_pos.deltaX), round(self.current_pos.deltaY), '.b')
                plt.pause(self.delta_t)
        if (self.current_pos - self.goal).length <= self.goal_threshold:
            self.is_path_plan_success = True
            if self.is_plot:
                plt.show()

    def calculate_path_length(self):
        """
        计算并返回路径的总长度
        """
        if len(self.path) < 2:
            return 0
        total_length = 0
        # 转换路径中的点为Vector2d对象，以便使用Vector2d的减法和length属性
        vector_path = [Vector2d(point[0], point[1]) for point in self.path]
        for i in range(1, len(vector_path)):
            step_length = (vector_path[i] - vector_path[i - 1]).length
            total_length += step_length
        t = total_length / v_ship
        return total_length, t


if __name__ == '__main__':
    # 相关参数设置
    k_att, k_rep = 1.0, 0.8
    rr = 3
    step_size, max_iters, goal_threshold = .5, 1000, .5  # 步长0.5寻路1000次用时4.37s, 步长0.1寻路1000次用时21s
    step_size_ = 2
    path_ = []

    # 设置速度
    v_ship = 5  # km/h

    # 设置、绘制起点终点
    start, goal = (28, 28), tuple(map(int, input("输入终点:").split(",")))
    is_plot = True
    if is_plot:
        fig = plt.figure(figsize=(7, 7))
        subplot = fig.add_subplot(111)
        subplot.set_xlabel('X-distance: m')
        subplot.set_ylabel('Y-distance: m')
        subplot.plot(start[0], start[1], '*r')
        subplot.plot(goal[0], goal[1], '*r')
    # 障碍物设置及绘制
    obs = [[20, 35], [25, 35], [35, 32], [35, 35], [35, 45], [45, 40], [50, 30]]
    print('obstacles: {0}'.format(obs))
    for i in range(0):
        obs.append([random.uniform(2, goal[1] - 1), random.uniform(2, goal[1] - 1)])

    if is_plot:
        for OB in obs:
            circle = Circle(xy=(OB[0], OB[1]), radius=rr, alpha=0.3)
            subplot.add_patch(circle)
            subplot.plot(OB[0], OB[1], 'xk')
    # t1 = time.time()
    # for i in range(1000):

    # Path plan
    if is_plot:
        apf = APF(start, goal, obs, k_att, k_rep, rr, step_size, max_iters, goal_threshold, is_plot)
    else:
        apf = APF(start, goal, obs, k_att, k_rep, rr, step_size, max_iters, goal_threshold, is_plot)
    apf.path_plan()
    print(apf.iters)
    if apf.is_path_plan_success:
        path = apf.path
        i = int(step_size_ / step_size)
        while i < len(path):
            path_.append(path[i])
            i += int(step_size_ / step_size)

        if path_[-1] != path[-1]:  # 添加最后一个点
            path_.append(path[-1])
        print('planed Path points:{}'.format(path_))
        for j in range(len(path_)):
            subplot.plot(path_[j][0], path_[j][1], '*g')
        print('Path plan success')
    else:
        print('planed Path points:{}'.format(apf.path))
        print('Path plan failed')
    # t2 = time.time()
    # print('寻路1000次所用时间:{}, 寻路1次所用时间:{}'.format(t2-t1, (t2-t1)/1000))

    # 重规划
    while start != goal:
        start = tuple(map(int, input("输入当前位置:").split(",")))  # 重新设置起点
        if start == goal:
            break
        is_plot = True
        if is_plot:
            fig = plt.figure(figsize=(7, 7))
            subplot = fig.add_subplot(111)
            subplot.set_xlabel('X-distance: m')
            subplot.set_ylabel('Y-distance: m')
            subplot.plot(start[0], start[1], '*r')
            subplot.plot(goal[0], goal[1], '*r')
            # 障碍物设置及绘制
            a = int(input("输入更新波浪的总数："))
            obs = [[] for i in range(a)]
            print("输入更新波浪坐标：")
            for i in range(a):
                obs[i] = list(map(float, input().split(',')))
            print('obstacles: {0}'.format(obs))
            for i in range(0):
                obs.append([random.uniform(2, goal[1] - 1), random.uniform(2, goal[1] - 1)])

            if is_plot:
                for OB in obs:

                    circle = Circle(xy=(OB[0], OB[1]), radius=rr, alpha=0.3)
                    subplot.add_patch(circle)
                    subplot.plot(OB[0], OB[1], 'xk')
            # t1 = time.time()0,0

            # for i in range(1000):

            # Path plan
            path_.clear()
            if is_plot:
                apf = APF(start, goal, obs, k_att, k_rep, rr, step_size, max_iters, goal_threshold, is_plot)
            else:
                apf = APF(start, goal, obs, k_att, k_rep, rr, step_size, max_iters, goal_threshold, is_plot)
            apf.path_plan()
            print(apf.iters)
            if apf.is_path_plan_success:
                path = apf.path
                i = int(step_size_ / step_size)
                while i < len(path):
                    path_.append(path[i])
                    i += int(step_size_ / step_size)

                if path_[-1] != path[-1]:  # 添加最后一个点
                    path_.append(path[-1])
                print('planed Path points:{}'.format(path_))
                for j in range(len(path_)):
                    subplot.plot(path_[j][0], path_[j][1], '*g')
                print('Path plan success')
            else:
                print('planed Path points:{}'.format(apf.path))
                print('Path plan failed')
