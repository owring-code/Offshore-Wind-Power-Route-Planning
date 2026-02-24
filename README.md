# 🌊 基于海域划分的海上风电场动态路径规划
*(Offshore Wind Power Route Planning)*

[![Conference](https://img.shields.io/badge/Conference-MSCE%202025-blue.svg)](https://dl.acm.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3760023.3760081-green.svg)](https://doi.org/10.1145/3760023.3760081)
[![Python](https://img.shields.io/badge/Python-3.x-yellow.svg)]()

> 本仓库为 **MSCE 2025** (International Conference on Management Science and Computer Engineering) 录用论文的配套开源项目。

---

## 📄 论文信息

- **标题**: A Dynamic Path Planning Method considering Sea Area Division for Off-shore Wind Farm Maintenance
- **作者**: W. Ouyang, W. Xu, Q. Xi, et al.
- **会议**: MSCE 2025
- **DOI**: `10.1145/3760023.3760081`
- **关键词**: `Dynamic path planning` | `Sea area division` | `K-Nearest Neighbors Algorithm` | `D* Algorithm` | `Artificial Potential Field Algorithm`

---

## 💡 研究摘要 (Abstract)

针对海上风电场维护作业的往返路径优化问题，本文提出了一种**基于近海与远海海域划分的动态路径规划模型**。
* **海域划分**：结合 **K最近邻 (KNN)** 算法对海域进行智能划分。
* **近海规划**：在近海区使用 **D* 算法** 进行路径规划，有效规避静态/慢速障碍。
* **远海规划**：在远海区使用 **改进的人工势场法 (APF)** 进行动态路径规划，实时响应风浪变化。

通过大量仿真实验验证，该模型能够高度适应复杂多变的海域环境，并规划出最优的运维路线。

---

## ⚙️ 核心方法与架构

本项目的方法论分为三个关键阶段：

### 1. 🗺️ 智能海域划分
使用 **K-Nearest Neighbors (KNN)** 算法，综合考量障碍物坐标分布与风浪气象数据，将目标作业海域科学划分为**近海 (Nearshore)**与**远海 (Offshore)**两部分，为后续的分层规划奠定基础。

### 2. 🚀 分层路径规划
| 海域类型 | 适用算法 | 处理场景 | 优势说明 |
| :----: | :----: | :----: | :---- ;|
| **近海区** | **D* 算法** | 礁石、浮标等静态或缓慢移动障碍物 | 适合已知或半已知环境中的全局寻优与局部重规划 |
| **远海区** | **改进人工势场法** | 涌浪、极端风力等快速变化的动态障碍 | 动态响应能力强，可实现海况骤变下的实时避障 |

### 3. 🔗 路径拼接与全局优化
系统自动计算并对比各分段路径的物理长度与风险成本，无缝拼接近海与远海路线，最终输出全局最短且最安全的往返作业路径。

---

## 🔬 仿真实验与结果

### 实验环境设置
* **环境构建**：虚拟二维海域环境，设定起点坐标为 `(0,0)`，终点坐标为 `(50,50)`。
* **障碍物设定**：随机生成近海静态礁石群，以及远海基于有效波高的动态障碍区域。

### 实验结果
通过对比不同海域划分点下的路径总长度，本模型成功锁定了**最优划分点 `(25,28)`**，并在仿真中完美展示了应对突发障碍的动态重规划过程。

![Simulation Result](https://private-user-images.githubusercontent.com/133517516/531799723-ff9aa2e1-3647-4928-822c-9ace9e340f29.png)
*(图：基于本模型生成的动态避障路径规划仿真结果图)*

---

## 📚 引用本项目 (Citation)

如果您在研究中使用了本仓库的代码或参考了我们的论文，请使用以下格式进行引用：

```bibtex
Ouyang W, Xu W, Xi Q, et al. A Dynamic Path Planning Method considering Sea Area Division for Off-shore Wind Farm Maintenance[C]//Proceedings of the 2025 International Conference on Management Science and Computer Engineering. 2025: 364-368.
