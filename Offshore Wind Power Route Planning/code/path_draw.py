import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# 数据点，包括起点和终点
points = [[28, 28], [30, 30], [31, 31], [32, 33], [32, 34], [33, 37], [34, 38], [36, 39], [37, 40], [39, 41], [41, 43], [42, 44], [44, 45], [45, 46], [47, 48], [49, 49], [50, 50]]
# 障碍物设置及绘制
obs = [[20, 35], [25, 35], [35, 32], [35, 35], [35, 45], [45, 40], [50, 30]]
# 分离x和y坐标
x_values = [point[0] for point in points]
y_values = [point[1] for point in points]


# 绘制折线图
plt.plot(x_values, y_values, marker='*', color='g', label='Path')
# 单独绘制起点和终点
plt.plot(points[0][0], points[0][1], marker='o', color='g', label='start')  # 起点
plt.text(points[0][0], points[0][1]-1.2, f'({points[0][0]}, {points[0][1]})', fontsize=9, ha='center')
plt.plot(points[-1][0], points[-1][1], marker='o', color='y', label='end')  # 终点

for ob in obs:
    circle = Circle(xy=(ob[0], ob[1]), radius=3, alpha=0.3)
    plt.gca().add_patch(circle)
    plt.plot(ob[0], ob[1], 'xk')

list_start = [(25, 28), (28, 28), (28, 25), (22, 28)]
for x, y in list_start[:-1]:
    plt.scatter(x, y, c="r", marker="o")

x, y = list_start[-1]
plt.scatter(x, y, c="r", marker="o", label='list_end')  # 只有这个会显示在图例中


total_distance = np.sum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
print(f'Total distance of the polyline: {total_distance:.2f}km')
# 显示图例
plt.legend()
plt.legend(loc='upper left')
plt.legend(fontsize='15')
plt.show()

