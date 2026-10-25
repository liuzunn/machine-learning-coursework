# -*- coding: utf-8 -*-
"""Notebook 第 1 部分：标题、环境、数据探索与划分"""

CELLS = []

CELLS.append(("md", r"""
# 机器学习课程作业 2 · B 组
## 手写逻辑回归与高斯朴素贝叶斯的实现与对比

| 项目 | 内容 |
|---|---|
| 班级 | 信计 2301 |
| 组别 | B 组（逻辑回归 + 朴素贝叶斯） |
| 学号 | XXXXXXXXXX |
| 姓名 | XXX |
| 完成日期 | 2026 年 10 月 25 日 |
| 运行环境 | Windows + Anaconda（Python 3.13.9）、NumPy、scikit-learn、Matplotlib |

**本 notebook 的内容**

1. 数据探索与固定划分（第 1 节）
2. 手写逻辑回归：极大似然估计 + 全批量梯度下降（第 2 节）
3. 手写高斯朴素贝叶斯：方差下限保护 + 对数域计算（第 3 节）
4. 手写评估指标：混淆矩阵、精确率、召回率、F1、P-R 曲线、AP（第 4 节）
5. 实验结果与图表（第 5 节）
6. 结论与结果导出（第 6 节）

**约束说明**：全程**不调用** scikit-learn 的 `LogisticRegression`、`GaussianNB`；
仅借用 `load_breast_cancer` 读取数据、`train_test_split` 划分数据，
以及 `sklearn.metrics` 中的 `average_precision_score` / `roc_auc_score`
用于**交叉校验手写指标的正确性**（不是用于建模）。
"""))

CELLS.append(("md", r"""
## 0. 环境准备

固定随机种子并设置中文字体。文中所有图都由本 notebook 的代码直接生成并保存到 `figures/` 目录，
报告里插入的图片与这里的输出是同一批文件，保证「截图与代码输出一致」。
"""))

CELLS.append(("code", r'''import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---------- 版本信息（写进报告，便于复现）----------
import sklearn
print(f"Python      : {sys.version.split()[0]}")
print(f"NumPy       : {np.__version__}")
print(f"pandas      : {pd.__version__}")
print(f"Matplotlib  : {matplotlib.__version__}")
print(f"scikit-learn: {sklearn.__version__}")

# ---------- 中文字体与绘图风格 ----------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]   # 中文字体
plt.rcParams["axes.unicode_minus"] = False                        # 正确显示负号
plt.rcParams["figure.dpi"] = 120                                  # 屏幕显示分辨率
plt.rcParams["savefig.dpi"] = 300                                 # 导出分辨率
plt.rcParams["savefig.bbox"] = "tight"                            # 导出时裁掉多余白边
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.axisbelow"] = True

# ---------- 全局随机种子：保证多人重跑结果一致 ----------
SEED = 42
np.random.seed(SEED)

# ---------- 输出目录 ----------
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
print("\n图表输出目录：", os.path.abspath(FIG_DIR))
'''))

CELLS.append(("md", r"""
## 1. 问题描述与数据

### 1.1 任务

采用**逻辑回归**与**高斯朴素贝叶斯**完成 scikit-learn 内置乳腺癌数据
（`load_breast_cancer`）的**二分类**任务；两种模型都要求**手写实现**，
不允许调用 `sklearn.linear_model.LogisticRegression` 与
`sklearn.naive_bayes.GaussianNB`。

### 1.2 数据

| 项目 | 说明 |
|---|---|
| 样本量 | 569 |
| 特征数 | 30（全部为连续型数值特征） |
| 类别 | 2 类：malignant 恶性（标签 0，212 条）、benign 良性（标签 1，357 条） |
| 标签含义 | 由乳腺肿块细针穿刺（FNA）图像计算出的细胞核形态指标判定 |
| 缺失值 | 无 |

30 个特征由 10 个形态指标（radius、texture、perimeter、area、smoothness、
compactness、concavity、concave points、symmetry、fractal dimension）的
**均值（mean）、标准误（error）、最差值（worst）** 三组构成。
"""))

CELLS.append(("code", r'''from sklearn.datasets import load_breast_cancer

# 加载内置数据（本地 csv，无需联网）
data = load_breast_cancer()
X = data.data                      # (569, 30) 特征矩阵
y = data.target                    # (569,)    标签：0=恶性, 1=良性
feature_names = data.feature_names
target_names = data.target_names

print("特征矩阵 X 形状 :", X.shape)
print("标签向量 y 形状 :", y.shape)
print("类别名称        :", list(target_names), "（0=malignant 恶性, 1=benign 良性）")
print("前 10 个特征名  :", list(feature_names)[:10], "... 共", len(feature_names), "个")
print()
# 类别分布：检查是否类别不平衡
unique, counts = np.unique(y, return_counts=True)
print("类别分布：")
for c, n in zip(unique, counts):
    print("  标签 %d（%-9s）: %3d 条，占 %.2f%%" % (c, target_names[c], n, 100.0 * n / len(y)))
print("合计: %d 条" % len(y))
'''))

CELLS.append(("md", r"""
### 1.3 特征量纲检查：为什么逻辑回归必须先标准化

逻辑回归用梯度下降求解时，步长是否合适取决于目标函数的 **Lipschitz 常数**，
而这个常数被特征的量纲直接放大。乳腺癌数据的 30 个特征量纲差别极大，
例如 area_worst 的取值在 200～4000 之间，而 smoothness_mean 只有 0.05 左右。
下面用统计量把这个差异量化出来。
"""))

CELLS.append(("code", r'''# 逐列统计：均值、标准差、最小值、最大值
stats = pd.DataFrame({
    "mean": X.mean(axis=0),
    "std": X.std(axis=0),
    "min": X.min(axis=0),
    "max": X.max(axis=0),
}, index=feature_names)

print("量纲差异最大的 5 个特征（按标准差降序）：")
print(stats.sort_values("std", ascending=False).head(5).round(4).to_string())
print()
print("量纲最小的 5 个特征：")
print(stats.sort_values("std", ascending=True).head(5).round(6).to_string())

ratio = stats["std"].max() / stats["std"].min()
print("\n最大标准差 = %.3f，最小标准差 = %.6f，两者相差 %.0f 倍" % (stats["std"].max(), stats["std"].min(), ratio))
print("worst area 取值范围: [%.1f, %.1f]" % (stats.loc["worst area", "min"], stats.loc["worst area", "max"]))
print("mean smoothness 取值范围: [%.4f, %.4f]" % (stats.loc["mean smoothness", "min"], stats.loc["mean smoothness", "max"]))

# 缺失值与非有限值检查（朴素贝叶斯对 NaN 很敏感，必须先确认数据干净）
print("\n是否存在缺失值 :", bool(np.isnan(X).any()))
print("是否存在无穷值 :", bool(np.isinf(X).any()))
print("全部为有限值   :", bool(np.isfinite(X).all()))
'''))

CELLS.append(("md", r"""
### 1.4 特征相关性：为后面解释「朴素贝叶斯为何吃亏」埋伏笔

高斯朴素贝叶斯的核心假设是**在给定类别的条件下各特征相互独立**。
如果原始特征之间高度线性相关，这个假设就被违背。先把这个事实量化出来。
"""))

CELLS.append(("code", r'''# 计算 30x30 皮尔逊相关系数矩阵（只用于诊断，不参与建模）
corr = np.corrcoef(X, rowvar=False)

# 取上三角（不含对角线），找出相关性最强的特征对
pairs = []
for i in range(corr.shape[0]):
    for j in range(i + 1, corr.shape[1]):
        pairs.append((abs(corr[i, j]), i, j, corr[i, j]))
pairs.sort(reverse=True)

print("相关系数绝对值最大的 8 对特征：")
for a, i, j, r in pairs[:8]:
    print("  %-20s - %-20s  r = %+.4f" % (feature_names[i], feature_names[j], r))

n_strong = sum(1 for a, *_ in pairs if a >= 0.9)
print()
print("|r| >= 0.9 的特征对共 %d 对（共 %d 对），占比 %.1f%%" % (n_strong, len(pairs), 100.0 * n_strong / len(pairs)))
print("|r| >= 0.8 的特征对共 %d 对" % sum(1 for a, *_ in pairs if a >= 0.8))
'''))

CELLS.append(("md", r"""
### 1.5 训练集 / 测试集划分

报告要求写清划分方式与随机种子，这里固定为：

- 划分比例 **训练 : 测试 = 7 : 3**（398 : 171）
- **分层抽样**（`stratify=y`），保证两类比例在两个子集中一致
- 主实验随机种子 **`random_state = 42`**
- 第 5.4 节再换 10 个不同种子做稳定性实验

标准化器只能在**训练集**上拟合：测试集必须当作"将来才会见到的数据"。
"""))

CELLS.append(("code", r'''from sklearn.model_selection import train_test_split

# 固定划分：分层抽样 + 固定随机种子
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,        # 30% 作测试集
    random_state=SEED,    # 固定种子 42，保证可复现
    stratify=y,           # 按类别分层，避免两子集类别比例失衡
)

# 断言：形状与分层比例都要符合预期
assert X_train.shape == (398, 30) and X_test.shape == (171, 30), "划分形状不符"
print("训练集: X %s, y %s" % (X_train.shape, y_train.shape))
print("测试集: X %s, y %s" % (X_test.shape, y_test.shape))
print("训练集正类（良性）比例: %.4f  （%d/%d）" % (y_train.mean(), y_train.sum(), len(y_train)))
print("测试集正类（良性）比例: %.4f  （%d/%d）" % (y_test.mean(), y_test.sum(), len(y_test)))
print("两者相差 %.4f，分层抽样生效" % abs(y_train.mean() - y_test.mean()))
print("类别计数  训练集: 恶性 %d / 良性 %d" % (np.sum(y_train == 0), np.sum(y_train == 1)))
print("类别计数  测试集: 恶性 %d / 良性 %d" % (np.sum(y_test == 0), np.sum(y_test == 1)))
'''))
