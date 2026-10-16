# 机器学习 · 课程记录与实验

《机器学习》课程的课堂练习与实验一（线性回归）。

## 目录

| 路径 | 内容 |
| :--- | :--- |
| [`notes/`](./notes/) | 课堂练习与示例代码（Python 语言基础、常用库） |
| [`lab01-线性回归/`](./lab01-线性回归/) | 实验一：线性回归 —— 代码、报告与结果图 |

## 实验一：线性回归

用线性回归对体测数据建模，步骤：

1. 读入数据集，做基础统计与可视化
2. 划分训练 / 测试集
3. 拟合线性模型
4. 用测试集评估，画「预测值 vs 真实值」散点图
5. 单特征回归，观察 BMI 与各指标的关系

| 文件 | 说明 |
| :--- | :--- |
| [`实验1_线性回归.ipynb`](./lab01-线性回归/实验1_线性回归.ipynb) | 实验代码（Jupyter Notebook） |
| [`实验报告.docx`](./lab01-线性回归/实验报告.docx) / [`.md`](./lab01-线性回归/实验报告.md) | 实验报告 |
| [`figures/`](./lab01-线性回归/figures/) | 结果图：预测值vs真实值、BMI 单特征 |

### 结果

![预测值 vs 真实值](lab01-线性回归/figures/fig1_test_pred_vs_true.png)

*十特征模型：测试集真实值（横轴）与预测值（纵轴），红色虚线为 y=x 参考线。点越贴近虚线说明拟合越好。*

![BMI 单特征](lab01-线性回归/figures/fig2_bmi_one_feature.png)

*训练集默认预缩放 BMI 与目标的散点，以及一元最小二乘拟合直线。*

## 课堂练习

| 文件 | 内容 |
| :--- | :--- |
| `notes/课次02_Python语言基础_练习.ipynb` | Python 语言基础练习 |
| `notes/课次02_示例代码.py` | 配套示例代码 |
| `notes/课次03_课堂示例.ipynb` / `.py` | Python 常用库（NumPy / Pandas / Matplotlib）示例 |
| `notes/课次03_练习.ipynb` | 常用库练习 |

## 环境

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## 许可

本仓库为个人课程学习记录，仅含本人完成的练习与实验代码。
