import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
plt.rcParams['font.size'] = 15
# 数据点，包括起点和终点
points = [[50, 50]]
# 障碍物设置及绘制
obs = [[20, 35], [25, 35], [35, 32], [35, 38], [35, 45], [45, 43], [50, 30]]
# 分离x和y坐标
x_values = [point[0] for point in points]
y_values = [point[1] for point in points]

# 绘制第一条折线
# plt.plot(x_values, y_values, marker='*', color='b', label='pre_path')

# 绘制障碍物
for ob in obs:
    circle = Circle(xy=(ob[0], ob[1]), radius=3, alpha=0.3)
    plt.gca().add_patch(circle)
    plt.plot(ob[0], ob[1], 'xk')

# 第二条折线的数据点
list_path = [[25, 28], [27, 30], [28, 31], [30, 32], [31, 34], [33, 35], [34, 35], [36, 35], [38, 37], [39, 39], [41, 40], [42, 41], [43, 44], [44, 46], [46, 47], [47, 48], [49, 49], [50, 50]]
list_x_values = [point[0] for point in list_path]
list_y_values = [point[1] for point in list_path]

# 绘制第二条折线
plt.plot(list_x_values, list_y_values, marker='*', color='g', label='final_path')

# 单独绘制起点和终点
plt.plot(points[0][0], points[0][1], marker='o', color='g', label='start')  # 起点
plt.text(points[0][0], points[0][1]+1, f'({points[0][0]}, {points[0][1]})', fontsize=12, ha='center')
plt.plot(points[-1][0], points[-1][1], marker='o', color='y', label='end')  # 终点

# 计算并显示总距离
total_distance = np.sum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
print(f'Total distance of the polyline: {total_distance:.2f}km')

# 显示图例
plt.legend()
plt.legend(fontsize='16')
plt.show()

