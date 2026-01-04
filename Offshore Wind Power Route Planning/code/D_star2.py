import math
import random
from sys import maxsize  # 导入最大数，2^63-1
import numpy as np

from matplotlib import pyplot as plt


class State(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."
        self.t = "new"
        self.h = 0
        self.k = 0  # k即为f

    def cost(self, state):
        if self.state == "#" or state.state == "#":
            return maxsize  # 存在障碍物时，距离无穷大
        return math.sqrt(math.pow((self.x - state.x), 2) +
                         math.pow((self.y - state.y), 2))

    def set_state(self, state):
        if state not in ["S", ".", "#", "E", "*", "+"]:
            return
        self.state = state


class Map(object):
    """
    创建地图
    """

    def __init__(self, row, col):
        self.row = row  # x轴
        self.col = col  # y轴
        self.map = self.init_map()

    def init_map(self):
        # 初始化map
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    def print_map(self):
        for i in range(self.row):
            tmp = ""
            for j in range(self.col):
                tmp += self.map[i][j].state + " "
            print(tmp)

    def plot_map(self):
        SE_state_list = []
        OB_state_list = []
        path_1 = []
        path_2 = []
        for i in range(self.row):
            for j in range(self.col):
                if self.map[i][j].state in ["S", "E"]:
                    SE_state_list.append([i, j])
                if self.map[i][j].state == "#":
                    OB_state_list.append([i, j])
                if self.map[i][j].state == "+":
                    path_1.append([i, j])
                if self.map[i][j].state == "*":
                    path_2.append([i, j])

        SE_state_list = np.array(SE_state_list)
        OB_state_list = np.array(OB_state_list)
        path_1 = np.array(path_1)
        path_2 = np.array(path_2)

        x0 = 0
        y0 = 0
        plt.scatter(x0, y0, c="g", marker="o", label='start', s=50)

        x_ = 28
        y_ = 25
        x1, y1 = SE_state_list[:, 0], SE_state_list[:, 1]
        plt.scatter(x1, y1, c="y", marker="o", label='end', s=50)
        plt.text(x_, y_-2, f'({x_}, {y_})', fontsize=15, ha='center')
        x2, y2 = OB_state_list[:, 0], OB_state_list[:, 1]
        plt.scatter(x2, y2, c="b", marker="*", s=50)
        list_start = [(25, 28), (22, 28), (28, 28)]
        for x, y in list_start[:-1]:
            plt.scatter(x, y, c="r", marker="o", s=50)

        x, y = list_start[-1]
        plt.scatter(x, y, c="r", marker="o", label='list_end')  # 只有这个会显示在图例中
        if len(path_1) != 0:
            x3, y3 = path_1[:, 0], path_1[:, 1]
            plt.plot(x3, y3, 'r*--', alpha=0.5, label="Path_first")
        if len(path_2) != 0:
            x4, y4 = path_2[:, 0], path_2[:, 1]
            plt.plot(x4, y4, 'g*--', alpha=0.5, label="Path")



    def get_neighbors(self, state):
        # 获取8邻域
        state_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if state.x + i < 0 or state.x + i >= self.row:
                    continue
                if state.y + j < 0 or state.y + j >= self.col:
                    continue
                state_list.append(self.map[state.x + i][state.y + j])
        return state_list

    def set_reef(self, point_list):
        # 设置礁石的位置
        for x, y in point_list:
            if x < 0 or x >= self.row or y < 0 or y >= self.col:
                continue
            self.map[x][y].set_state("#")


def calculate_path_length(Start, End):
    length = 0.0
    s = Start.parent
    while s != End:
        if s.parent:
            length += s.cost(s.parent)
            s = s.parent
    print(f"规划完成的路径长度为：{length:.2f}km")


class Dstar(object):

    def __init__(self, maps):
        self.map = maps
        self.open_list = set()  # 创建空集合

    def process_state(self):
        """
        D*算法的主要过程
        :return:
        """
        x = self.min_state()  # 获取open list列表中最小k的节点
        if x is None:
            return -1
        k_old = self.get_kmin()  # 获取open list列表中最小k节点的k值
        self.remove(x)  # 从open list中移除
        # 判断open list中
        if k_old < x.h:
            for y in self.map.get_neighbors(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
        if k_old == x.h:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or (y.parent == x and y.h != x.h + x.cost(y)) \
                        or (y.parent != x and y.h > x.h + x.cost(y)):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or (y.parent == x and y.h != x.h + x.cost(y)):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                else:
                    if y.parent != x and y.h > x.h + x.cost(y):
                        self.insert(x, x.h)
                    else:
                        if y.parent != x and x.h > y.h + x.cost(y) \
                                and y.t == "close" and y.h > k_old:
                            self.insert(y, y.h)
        return self.get_kmin()

    def min_state(self):
        if not self.open_list:
            return None
        min_state = min(self.open_list, key=lambda x: x.k)  # 获取open list中k值最小对应的节点
        return min_state

    def get_kmin(self):
        # 获取open list表中k(f)值最小的k
        if not self.open_list:
            return -1
        k_min = min([x.k for x in self.open_list])
        return k_min

    def insert(self, state, h_new):
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)

    def remove(self, state):
        if state.t == "open":
            state.t = "close"
        self.open_list.remove(state)

    def modify_cost(self, x):
        if x.t == "close":  # 是以一个open list，通过parent递推整条路径上的cost
            self.insert(x, x.parent.h + x.cost(x.parent))

    def run(self, Start, End, List):
        self.insert(End, 0)
        while True:
            self.process_state()
            if Start.t == "close":
                break
        Start.set_state("S")
        s = Start
        while s != End:
            s = s.parent
            s.set_state("+")
        s.set_state("E")
        tmp = Start

        # 起始点不变

        '''
        从起始点开始，往目标点行进，当遇到障碍物时，重新修改代价，再寻找路径
        '''
        while tmp != End:
            tmp.set_state("*")
            # self.map.print_map()
            # print("")
            if tmp.parent.state == "#":
                self.modify(tmp)
                continue
            tmp = tmp.parent
        tmp.set_state("E")
        print('障碍物发生变化时，搜索的路径如下(*为更新的路径)：')
        self.map.print_map()
        self.map.plot_map()


    def modify(self, state):

        """
        当障碍物发生变化时，从目标点往起始点回推，更新由于障碍物发生变化而引起的路径代价的变化
        :param state:
        :return:
        """

        self.modify_cost(state)
        while True:
            k_min = self.process_state()
            if k_min >= state.h:
                break


if __name__ == '__main__':
    plt.rcParams['font.size'] = 15
    m = Map(40, 40)

    the_list = [(4, 7), (10, 10), (17, 5), (16, 9), (7, 15), (3, 25), (18, 20), (25, 15), (20, 15)]
    m.set_reef(the_list)  # 已知障碍物的位置

    start = m.map[0][0]
    end = m.map[28][25]
    dstar = Dstar(m)
    dstar.run(start, end, the_list)
    calculate_path_length(start, end)
    plt.legend(fontsize='13')
    plt.show()
    # m.print_map()



