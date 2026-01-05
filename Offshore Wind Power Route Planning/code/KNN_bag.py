import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
from itertools import product


def load_data():
    n = int(input("输入礁石与波浪的总数："))
    Data = [[] for _ in range(n)]
    for i in range(n):
        Data[i] = list(map(float, input().split(',')))  # 坐标值
    Data = np.array(Data)

    list1 = input("对应标签值:").split()  # 标签值
    Labels = []
    for j in range(n):
        Labels.append(list1[j])

    return Data, Labels

def plot_data(Data, Labels):
    # 解析出每个样本点的x1特征和x2特征
    x1, x2 = Data[:, 0], Data[:, 1]
    plt.scatter(x1, x2, label='obstacle')

    # 在图中根据每个样本的类别，标注上对应的文字说明
    for i, xy in enumerate(zip(x1, x2)):
        plt.annotate(Labels[i],  # 从标签列表中找到对应的类别文字作为标注
                     xy=xy,  # 指定要标注的位置的坐标
                     xytext=(-15, -4),  # 标注的文字与指定位置的位移
                     textcoords='offset points')

class KNN:
    def __init__(self, k=3):
        self.k = k  # 需要依据的距离最近的多少个样本的类别来进行分类预测
        self.data_size = 0
        self.data = data
        self.labels = labels

    def fit(self, Data, Labels):  # 实现模型训练，数据的拟合
        # 在knn中，不需要先进行大量的训练和数据拟合，只需将数据记录下来，留待测时计算距离即可
        self.data_size = Data.shape[0]
        self.data = Data
        self.labels = Labels

    def predict(self, sample):  # 对待预测的样本进行类别预测
        # 计算待预测样本与训练集中的每个样本的欧氏距离
        dists = np.sqrt(np.sum((np.tile(sample, (self.data_size, 1)) - data) ** 2, axis=1))
        # 按照距离进行排序
        sorted_indices = np.argsort(dists)
        label_count = {}
        # 循环距离索引列表中的前k个样本即可
        for i in range(self.k):
            voted_label = labels[sorted_indices[i]]  # 获取最近的k个样本对应的真实类别
            # 给各个类别分别计数
            label_count[voted_label] = label_count.get(voted_label, 0) + 1
            # 接下来需要少数服从多数
        sorted_label_count = sorted(label_count.items(),
                                    key=lambda x: x[1],
                                    reverse=True)
        # 返回出现次数最多的那个类别，作为我们的预测类别
        return sorted_label_count[0][0]

def clear_redundant_legend():
    # 迭代地将当前plt所有的handles和labels返回到两个变量中
    Handles, Labels = plt.gca().get_legend_handles_labels()
    # 去除冗余的图例，利用字典的特性
    by_label = OrderedDict(zip(Labels, Handles))
    Handles, Labels = by_label.values(), by_label.keys()

    return Handles, Labels

def solution(x, y, n):
    test1 = []
    list1 = []
    for i in range(1, n):
        for z in product(range(x-i, x+i+1), range(y-i, y+i+1)):
            test1.append(z)
        for t in test1:
            set_label = model.predict(t)
            if set_label == 'B':
                list1.append(set_label)
        if not list1:
            test1.clear()
        else:
            break
    return test1



if __name__ == '__main__':
    plt.rcParams['font.size'] = 13
    # 标注起点
    start, goal = (0, 0), (15.0, 15.0)
    plt.scatter(start[0], start[1], marker='o', color='g', label='start', s=25)
    plt.scatter(goal[0], goal[1], marker='o', color='y', label='goal', s=25)

    data, labels = load_data()
    plot_data(data, labels)
    # 以类的形式创建一个knn对象， 用这个模型对象进行训练和预测
    model = KNN(k=3)
    # 进行模型训练，实现数据的拟合
    model.fit(data, labels)

    # 执行knn对新样本的预测
    samples = solution(int((goal[0]-start[0])/2), int((goal[1]-start[1])/2),
                       int(max((goal[0]-start[0]), (goal[1]-start[1]))))
    # 执行knn对新样本的预测
    list_ = []
    for s in samples:
        plt.scatter(s[0], s[1], marker='x', color='magenta', label='predict', s=25)
        pred_label = model.predict(s)
        if pred_label == 'B':
            list_.append(s)
        plt.annotate(pred_label,
                     xy=s,
                     xytext=(-15, -4),
                     textcoords='offset points')
    handles, labels = clear_redundant_legend()
    # legend 接收两个参数
    print(list_)
    plt.legend(handles, labels, fontsize=15)
    plt.show()
