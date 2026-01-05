# 论文信息
标题：A Dynamic Path Planning Method considering Sea Area Division for Off-shore Wind Farm Maintenance

会议：MSCE 2025 (International Conference on Maritime Science and Engineering)

DOI：10.1145/3760023.3760081

## 摘要
本文提出一种基于近海与远海海域划分的动态路径规划模型，用于优化海上风电场维护作业的往返路径。该模型结合K最近邻算法对海域进行划分，并在近海区使用D*算法规划路径，在远海区使用改进的人工势场法进行动态路径规划。通过仿真实验验证，该模型能够适应复杂的海域环境并规划出最优维护路径。

## 关键词
Dynamic path planning, Sea area division, K-Nearest Neighbors Algorithhm, D* Algorithm, Artificial Potential Field Algorithm

## 主要方法概述
1、海域划分：使用K最近邻算法基于障碍物坐标、风浪数据将海域划分为近海与远海两部分。

2、路径规划：

近海区：采用D*算法规划，适用于静态或移动障碍物（如礁石、浮标）。

远海区：采用改进的人工势场法，动态响应风浪变化，实现实时避障。

3、路径拼接与优化：通过计算各分段路径长度，选择最短往返路径。

## 仿真实验
* 数据集：虚拟海域环境，设定起点(0,0)、终点(50,50)，随机生成近海礁石与远海有效波高障碍。

* 结果：通过对比不同划分点下的路径总长度，确定最优划分点(25,28)，并展示了动态重规划过程。

<img width="1514" height="749" alt="image" src="https://github.com/user-attachments/assets/ff9aa2e1-3647-4928-822c-9ace9e340f29" />


## 引用格式
建议引用格式：
~~~text
Ouyang W, Xu W, Xi Q, et al. A Dynamic Path Planning Method considering Sea Area Division for Off-shore Wind Farm Maintenance[C]//Proceedings of the 2025 International Conference on Management Science and Computer Engineering. 2025: 364-368.
~~~
