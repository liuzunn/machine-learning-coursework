# -*- coding: utf-8 -*-
"""生成 Word 报告：B_XXXXXXXXXX_XXX_作业2_报告.docx"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import Cm, Pt

import nbformat
from docx_helpers import (BODY_FONT, HEAD_FONT, blank, bullet, caption_below,
                          code_block, equation, figure, h1, h2, h3, page_break,
                          para, set_run_font, setup_document, table)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB = os.path.join(ROOT, "B_XXXXXXXXXX_XXX_作业2_代码.ipynb")
OUT = os.path.join(ROOT, "B_XXXXXXXXXX_XXX_作业2_报告.docx")
FIG = lambda n: os.path.join(ROOT, "figures", n)

R = json.load(open(os.path.join(ROOT, "results.json"), encoding="utf-8"))
LR, NB_, META = R["lr"], R["gnb"], R["meta"]
E_LR, E_NB = R["eval"]["lr"], R["eval"]["gnb"]


def f4(x):
    return "%.4f" % float(x)


def pc(x):
    return "%.2f%%" % (100.0 * float(x))


def cell_source(needle):
    """从已执行的 notebook 中按关键字取出代码单元格源码，保证报告与代码一致"""
    nb = nbformat.read(NB, as_version=4)
    for c in nb.cells:
        if c.cell_type == "code" and needle in c.source:
            return c.source
    raise KeyError(needle)


# ============================================================ 封面
def build_cover(doc):
    blank(doc, 2)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    set_run_font(p.add_run("    理学院"), HEAD_FONT, 16)

    blank(doc, 3)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    set_run_font(p.add_run("机器学习课程作业报告"), HEAD_FONT, 26, bold=True)

    blank(doc, 2)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    set_run_font(p.add_run("作业 2：手写实现与对比实验"), HEAD_FONT, 18)

    blank(doc, 2)
    for line in ["B 组　逻辑回归与高斯朴素贝叶斯的", "手写实现与对比"]:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        set_run_font(p.add_run(line), HEAD_FONT, 15)

    blank(doc, 5)
    for label, value in [("班　　级", "信计 2301"),
                         ("组　　别", "B 组"),
                         ("学　　号", "XXXXXXXXXX"),
                         ("姓　　名", "刘　尊"),
                         ("完成日期", "2026 年 10 月 25 日")]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run("%s：　%s" % (label, value))
        set_run_font(r, BODY_FONT, 14)
    page_break(doc)


# ============================================================ 一、问题描述
def build_section1(doc):
    h1(doc, "一、问题描述")

    h2(doc, "1.1　研究任务")
    para(doc, "本文要解决的问题是：在 scikit-learn 内置的乳腺癌数据集"
              "（load_breast_cancer）上完成二分类，并对两种手写实现的概率模型——"
              "**逻辑回归**与**高斯朴素贝叶斯**——做系统的实验对比。按作业要求，"
              "两种模型都不允许调用 sklearn 的 LogisticRegression 与 GaussianNB，"
              "全部由 NumPy 从零实现；求解过程、数值稳定性处理与参数估计的每一步"
              "都由自己完成，代码见随本报告提交的 notebook。")

    h2(doc, "1.2　数据集")
    para(doc, "数据共 %d 条样本、%d 个连续特征、%d 个类别（恶性 %d 条、良性 %d 条）。"
              "30 个特征由 10 个细胞核形态指标的**均值、标准误、最差值**三组构成，"
              "10 个指标分别是半径（radius）、纹理（texture）、周长（perimeter）、"
              "面积（area）、平滑度（smoothness）、紧凑度（compactness）、凹度（concavity）、"
              "凹点（concave points）、对称性（symmetry）和分形维数（fractal dimension）。"
              "标签由细针穿刺（FNA）图像的病理诊断给出：**0 表示恶性（malignant）、"
              "1 表示良性（benign）**。数据中没有缺失值，也没有非有限值。数据概况见表 1。"
              % (META["n_samples"], META["n_features"], 2,
                 META["class_counts"]["malignant(0)"], META["class_counts"]["benign(1)"]))
    table(doc,
          ["项目", "内容"],
          [["样本量", "%d 条" % META["n_samples"]],
           ["特征数", "%d 个，全部为连续型数值特征" % META["n_features"]],
           ["类别分布", "恶性（0）%d 条（%.1f%%）；良性（1）%d 条（%.1f%%）"
            % (META["class_counts"]["malignant(0)"],
               100.0 * META["class_counts"]["malignant(0)"] / META["n_samples"],
               META["class_counts"]["benign(1)"],
               100.0 * META["class_counts"]["benign(1)"] / META["n_samples"])],
           ["缺失值", "无（已用 np.isnan / np.isfinite 校验）"],
           ["训练/测试划分", "7 : 3 分层抽样，训练 %d 条、测试 %d 条，random_state = %d"
            % (META["split"]["train"], META["split"]["test"], META["seed"])],
           ["评价指标", "精确率、召回率、F1、AP（均取宏平均）与 AUC、ECE"]],
          "表 1　数据集概况与实验设置",
          col_widths=[4.0, 11.2], align_first_left=True)

    h2(doc, "1.3　数据特点与建模难点")
    bullet(doc, "（1）", "**特征量纲差异极大。** 30 个特征的标准差最大为 %.3f、最小为 %.6f，"
                        "相差 %.0f 倍。这对逻辑回归的梯度下降是致命的：步长被最大特征尺度"
                        "卡死，小尺度特征的更新几乎停滞，收敛极慢（2.1 节的 Lipschitz 常数与"
                        "4.9 节的消融实验都会量化这一点）。"
           % (568.856, 0.002644, META["std_ratio_max_min"]))
    bullet(doc, "（2）", "**特征之间高度线性相关。** 30 个特征共有 %d 对两两组合，其中 "
                        "|r| ≥ 0.9 的有 %d 对。例如 mean radius、mean perimeter、mean area "
                        "三者本质上是同一几何量的不同度量，相关系数接近 0.99。"
                        "这一事实直接违背了朴素贝叶斯的条件独立假设。"
           % (30 * 29 // 2, META["n_strong_corr_pairs_0.9"]))
    bullet(doc, "（3）", "**类别不均衡。** 恶性 : 良性 ≈ %d : %d，正类占 %.1f%%。"
                        "若只看准确率，把样本全判为良性就能拿到 %.1f%%，因此必须用宏平均与"
                        "AP 这类对类别不均衡不敏感的指标。"
           % (META["class_counts"]["malignant(0)"], META["class_counts"]["benign(1)"],
              100.0 * META["class_counts"]["benign(1)"] / META["n_samples"],
              100.0 * META["class_counts"]["benign(1)"] / META["n_samples"]))
    bullet(doc, "（4）", "**数据在训练集上近似线性可分。** 30 维空间里存在一个超平面把几乎"
                        "所有训练样本正确分开。这带来一个理论后果：**不带正则的逻辑回归，"
                        "极大似然解不存在**——损失可以无限逼近 0，但最优权重发散到无穷。"
                        "这是本文必须引入 L2 正则的根本原因（2.1.6 节）。")
    bullet(doc, "（5）", "**样本量相对特征数并不宽裕。** 398 条训练样本对应 30 个特征，"
                        "特征数与样本数之比约 1 : 13。模型的方差控制因此变得重要。")

    h2(doc, "1.4　评价指标及其选择理由")
    para(doc, "精确率（Precision）衡量“判为恶性的样本中有多少真的是恶性”，"
              "召回率（Recall）衡量“真正的恶性样本有多少被找出来”，两者此消彼长，"
              "F1 是二者的调和平均。仅看单点 F1 会依赖“阈值取 0.5”这个人为选择，"
              "因此本文以 **AP（Average Precision）** 为主要指标：它等于 P-R 曲线下的面积，"
              "把所有阈值下的表现都考虑进去，衡量的是**概率排序质量**，"
              "与阈值无关。")
    para(doc, "考虑到两类样本量之比为 212 : 357，本文统一采用**宏平均**（macro）："
              "先对每个类别单独算指标，再取算术平均，两类等权。同时给出按样本数加权的"
              "加权平均作为补充，两者对照可以反映类别不均衡的影响。此外补充 AUC 与 "
              "ECE（期望校准误差），分别刻画排序能力与概率校准质量。")
    para(doc, "从医疗应用的角度看，**漏诊（把恶性判成良性）的代价远高于误诊**，"
              "所以真正部署时应当使用比 0.5 更低的阈值以换取更高的召回率，"
              "4.5 节的阈值分析给出了完整的取舍曲线。")

    h2(doc, "1.5　实验设置与可复现性")
    para(doc, "训练集与测试集按 7 : 3 分层划分，随机种子固定为 %d，其余所有随机性"
              "（如 5 折交叉验证的折划分）也全部固定种子。超参数只允许在**训练集内部**"
              "用交叉验证选择，测试集在超参数确定之前完全不参与任何计算。运行环境为："
              "Python %s、NumPy %s、scikit-learn %s、Matplotlib %s。"
              % (META["seed"], META["versions"]["python"], META["versions"]["numpy"],
                 META["versions"]["sklearn"], META["versions"]["matplotlib"]))


# ============================================================ 二（续）：2.2.6 / 2.2.7
def build_section2b(doc):
    h3(doc, "2.2.6　尺度不变性：为什么朴素贝叶斯不需要标准化")
    para(doc, "这是一个容易被忽略、但对理解两种方法差异很重要的性质。设对第 j 个特征做仿射变换"
              "x′_j = a_j·x_j + b_j（a_j > 0），则类条件均值与标准差同步变换为 "
              "μ′ = a_j·μ + b_j、σ′ = a_j·σ。由一维随机变量的密度变换公式，变换后的对数密度满足")
    equation(doc, r"\ln P(x'_{j} \mid y=c) = -\ln a_{j} + \ln P(x_{j} \mid y=c)", "16")
    para(doc, "即每个特征只多出一个常数 −ln a_j，而**这个常数与类别 c 无关**。"
              "对 30 个特征求和，得到 s′_c = s_c − Σ ln a_j。由于 MAP 决策只比较不同类别之间"
              "s_c 的相对大小，所有类别同时减去同一个常数不改变 argmax，也不改变归一化后的后验概率。"
              "所以对高斯朴素贝叶斯做逐特征标准化，**结果严格不变**（实测后验概率最大差 3.57e-14，"
              "属于浮点舍入）。这与逻辑回归形成鲜明对比：后者不做标准化就几乎无法用梯度下降训练。"
              "需要注意的是，取对数这类**非线性**变换不满足上述推导，会改变结果。")

    h3(doc, "2.2.7　程序流程图")
    para(doc, "图 2 给出高斯朴素贝叶斯的训练与预测流程。上半部分是训练：按类别分组统计先验、"
              "均值与方差（步骤 1、2），做方差下限保护（步骤 3），预计算对数先验与对数归一化常数"
              "（步骤 4）。下半部分是预测：对每个测试样本计算未归一化的对数后验 s_c（步骤 5），"
              "用 log-sum-exp 归一化成概率（步骤 6），取最大值作为预测并输出该类概率（步骤 7）；"
              "菱形判定控制对所有测试样本的循环。整个训练过程没有任何迭代。")
    figure(doc, FIG("fig2_gnb_flowchart.png"), "图 2　高斯朴素贝叶斯训练与预测流程图", width_cm=13.2)


# ============================================================ 三、算法实现
def build_section3(doc):
    h1(doc, "三、算法实现")
    para(doc, "全部代码写在随报告提交的 notebook 中（共 33 个代码单元，保留全部运行输出，"
              "零报错）。本节先说明代码结构，再贴出核心代码段。代码中的注释逐行说明每一句在做什么，"
              "与 notebook 中的代码完全一致。")

    h2(doc, "3.1　代码结构与函数划分")
    para(doc, "整个实现分成四层：数值稳定工具函数 → 手写标准化器 → 两个模型类 → 手写评估指标。"
              "这种分层的好处是每一层都可以独立测试，出问题时可快速定位。主要函数见表 2。")
    table(doc,
          ["层次", "函数 / 方法", "作用说明"],
          [["数值稳定", "sigmoid / log_sigmoid / logsumexp",
            "分段实现 sigmoid；用 logaddexp 算 ln σ(z)；log-sum-exp 防上下溢"],
           ["预处理", "StandardScalerFromScratch.fit / transform",
            "均值与标准差只在训练集估计，测试集只做变换"],
           ["逻辑回归", "LogisticRegressionGD._loss_grad",
            "一轮迭代内同时算出损失、梯度与概率 p"],
           ["逻辑回归", "LogisticRegressionGD.lipschitz",
            "由 X̃ᵀX̃/N 的最大特征值给出理论步长 1/L"],
           ["逻辑回归", "LogisticRegressionGD.fit",
            "全批量梯度下降主循环，三重收敛判据"],
           ["逻辑回归", "LogisticRegressionGD.fit_irls",
            "牛顿法解同一个正则化 MLE，用于校验收敛"],
           ["逻辑回归", "LogisticRegressionGD.predict_proba / predict",
            "输出 σ(z) 组成的 (N,2) 概率矩阵与 0.5 阈值判别"],
           ["朴素贝叶斯", "GaussianNBFromScratch.fit",
            "闭式估计先验、类条件均值与方差，并做方差下限保护"],
           ["朴素贝叶斯", "GaussianNBFromScratch.joint_log_likelihood",
            "两次矩阵乘法算出 (N,2) 对数联合概率"],
           ["朴素贝叶斯", "GaussianNBFromScratch.predict_proba / predict",
            "log-sum-exp 归一化成后验概率，取 argmax 作为预测"],
           ["评估", "confusion_matrix / binary_prf / prf_report",
            "混淆矩阵、逐类精确率/召回率/F1、宏平均与加权平均"],
           ["评估", "pr_curve / average_precision",
            "P-R 曲线与阶梯积分形式的 AP"],
           ["评估", "roc_auc / ece_score",
            "秩和公式计算 AUC，分箱计算期望校准误差"]],
          "表 2　主要函数与方法清单",
          col_widths=[2.4, 7.0, 6.0], align_first_left=True, font_size=9.0)

    h2(doc, "3.2　逻辑回归核心代码")
    para(doc, "首先实现三个数值稳定函数。sigmoid 按符号分段，log_sigmoid 用 logaddexp 实现式 (8)，"
              "logsumexp 用减最大值的方式避免上下溢。")
    code_block(doc, cell_source("def sigmoid(z):").split("# ---------- 自检")[0].rstrip())

    para(doc, "训练前必须先做 z-score 标准化，而标准化的均值与标准差只能在训练集上估计，"
              "否则测试集的信息会漏进模型。下面这个手写标准化器只有十几行，但它是逻辑回归能收敛的前提。")
    code_block(doc, cell_source("class StandardScalerFromScratch:").split("# 自检")[0].rstrip())

    para(doc, "下面是逻辑回归类的主体。要点：① 标准化器只在 fit 阶段用训练集拟合；"
              "② 增广矩阵把偏置并入权重，梯度公式与 Hessian 都不必为截距单独处理；"
              "③ 每轮迭代同时算出损失、梯度与概率，避免重复计算；"
              "④ fit_irls 用牛顿法解同一个正则化 MLE，用来验证梯度下降确实收敛到了最优解。")
    code_block(doc, cell_source("class LogisticRegressionGD:").split('print("LogisticRegressionGD')[0].rstrip())

    h2(doc, "3.3　高斯朴素贝叶斯核心代码")
    para(doc, "朴素贝叶斯没有迭代过程，核心在 fit 里的参数估计与 joint_log_likelihood 里的"
              "矩阵化实现。方差下限保护在 fit 中一行 max 完成；预测时先算未归一化的对数后验，"
              "再做 log-sum-exp 归一化。")
    code_block(doc, cell_source("class GaussianNBFromScratch:").split('print("GaussianNBFromScratch')[0].rstrip())

    h2(doc, "3.4　评估指标的实现")
    para(doc, "AP 的定义是 P-R 曲线下的阶梯面积。实现上有三个容易出错的地方："
              "① 只在**分数变化处**取阈值点，否则并列分数会把同一条水平线重复计入；"
              "② P-R 曲线的起点要补 (R=0, P=1)；③ 用 R 的差分乘以 P 求和，而不是用梯形法则。"
              "下面的实现与 sklearn 的定义完全一致（3.5 节会做逐项校验）。")
    code_block(doc, cell_source("def confusion_matrix(").rsplit("\n\nprint(", 1)[0].rstrip())

    h2(doc, "3.5　实现正确性的三重验证")
    para(doc, "手写实现最怕“算错了但看起来很对”。本文用三种互相独立的方式验证。")
    bullet(doc, "（1）", "**梯度下降 vs 牛顿法。** fit_irls 用二阶方法（IRLS）解同一个带 L2 的 MLE。"
                         "在 λ = 1e-2 时，两种方法得到的权重最大绝对差仅 8.15e-4，"
                         "损失值在小数点后 6 位一致（0.096144），说明梯度下降确实收敛到了全局最优，"
                         "而不只是“损失不再下降”。")
    bullet(doc, "（2）", "**手写指标 vs sklearn.metrics。** 用 sklearn 的 average_precision_score、"
                         "roc_auc_score、precision_score 等对同一批概率重算，"
                         "12 项指标的最大绝对差为 1.11e-16（纯浮点舍入级别）。")
    bullet(doc, "（3）", "**理论性质的自检。** ① 标准化后训练集每列均值为 5.2e-15、标准差偏差 1.0e-15；"
                         "② 高斯朴素贝叶斯的尺度不变性给出后验概率最大差 3.57e-14；"
                         "③ 退化特征实验里，不加方差下限则后验必然出现 NaN，加上之后预测恢复 0.9357 的准确率。"
                         "这些断言都写在 notebook 里，任何一条不成立都会直接抛异常。")
    para(doc, "为方便对照，表 3 汇总了 notebook 中五处关键断言的输出。")
    table(doc,
          ["验证项", "对照基准", "实测差异 / 结果", "结论"],
          [["梯度下降的权重", "牛顿法（IRLS）解，λ=1e-2", "最大差 8.15e-4", "收敛到同一最优解"],
           ["手写 AP / AUC / P / R / F1", "sklearn.metrics", "最大绝对差 1.11e-16", "实现完全正确"],
           ["标准化后的列统计量", "均值 0、标准差 1", "5.2e-15 / 1.0e-15", "标准化正确"],
           ["朴素贝叶斯尺度不变性", "原始尺度 vs 标准化", "后验最大差 3.57e-14", "理论性质成立"],
           ["退化特征（无方差下限）", "有方差下限", "NaN vs 正常预测", "保护机制必要"]],
          "表 3　实现正确性的关键断言",
          col_widths=[4.0, 4.8, 3.6, 3.0], align_first_left=True, font_size=9.5)


# ============================================================ 二、算法设计
def build_section2(doc):
    h1(doc, "二、算法设计")
    para(doc, "本节按“先讲模型、再讲目标函数、最后讲求解”的顺序，说明两种方法的原理。"
              "逻辑回归是**判别式模型**：直接对条件概率 P(y|x) 建模，用极大似然估计参数；"
              "高斯朴素贝叶斯是**生成式模型**：对类条件密度 P(x|y) 与类先验 P(y) 建模，"
              "再用贝叶斯公式反推后验。两者的本质差异，正是第五节性能差异的根源。")

    # -------------------------------------------------------------- 2.1
    h2(doc, "2.1　逻辑回归（Logistic Regression）")

    h3(doc, "2.1.1　模型形式与概率解释")
    para(doc, "逻辑回归先用特征的线性组合得到一个实数得分，再用 sigmoid 函数把它压缩到 (0,1)，"
              "解释为样本属于正类（良性）的概率：")
    equation(doc, r"\hat{p} = \sigma(\mathbf{w}^{\top}\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^{\top}\mathbf{x} + b)}}", "1")
    para(doc, "将上式变形可得**对数几率（log-odds）是线性的**：", indent=True)
    equation(doc, r"\ln\frac{\hat{p}}{1 - \hat{p}} = \mathbf{w}^{\top}\mathbf{x} + b", "2")
    para(doc, "这解释了为什么逻辑回归是线性分类器：它假设“正负两类的对数几率之比”随特征线性变化，"
              "两个类别之间的**决策边界是超平面** w^T x + b = 0。与之相比，朴素贝叶斯虽然"
              "也能得到线性边界，但边界的位置由类条件密度的参数间接决定。")
    para(doc, "为了书写方便，把偏置 b 并入权重：在特征矩阵右侧补一列 1，记增广样本为 "
              "x̃ = [x, 1]、增广权重为 w = [w, b]，模型统一写成 p̂ = σ(w^T x̃)。"
              "后文的 X̃ 均指增广后的设计矩阵（形状 N×(d+1)）。")

    h3(doc, "2.1.2　目标函数：极大似然估计")
    para(doc, "把标签看作伯努利随机变量，单个样本的似然为 P(y|x) = p̂^y (1-p̂)^(1-y)。"
              "N 个样本相互独立时，总似然为")
    equation(doc, r"L(\mathbf{w}) = \prod_{i=1}^{N} \hat{p}_i^{\,y_i}\,(1 - \hat{p}_i)^{\,1 - y_i}", "3")
    para(doc, "连乘容易下溢且难以求导，取对数把连乘变成连加（对数函数单调递增，"
              "不改变极值点）。极大化对数似然等价于极小化**平均负对数似然**（即交叉熵），"
              "再加上 L2 正则项，最终的目标函数为")
    equation(doc, r"J(\mathbf{w}) = -\frac{1}{N}\sum_{i=1}^{N}\left[ y_i \ln \hat{p}_i + (1 - y_i)\ln(1 - \hat{p}_i) \right] + \frac{\lambda}{2}\sum_{j=1}^{d} w_j^{2}", "4")
    para(doc, "式中 λ ≥ 0 是正则系数，偏置 b 不参与正则（它只影响决策边界的平移，"
              "惩罚它没有意义）。J 由两部分组成：第一项衡量拟合程度（越小越拟合训练集），"
              "第二项惩罚大的权重（越大越平滑）。J 是 w 的**凸函数**"
              "（负对数似然为凸，L2 为严格凸），因此局部最优即全局最优。")

    h3(doc, "2.1.3　梯度与梯度下降")
    para(doc, "对 z_i = w^T x̃_i 用链式法则，注意 dσ/dz = σ(1-σ)，可得"
              "∂/∂z [y ln σ(z) + (1-y) ln(1-σ(z))] = y - σ(z)，进而")
    equation(doc, r"\nabla J(\mathbf{w}) = \frac{1}{N}\tilde{\mathbf{X}}^{\top}(\hat{\mathbf{p}} - \mathbf{y}) + \lambda\,\mathbf{w}_{1:d}", "5")
    para(doc, "其中 p̂ 是所有样本的预测概率向量。梯度有很直观的解释：每个样本对参数的"
              "贡献正比于“预测误差 (p̂_i - y_i)”乘以它的特征值，预测错得越多、"
              "特征值越大的样本，对参数更新的影响越大。"
              "用全批量梯度下降迭代求解：")
    equation(doc, r"\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta\,\nabla J(\mathbf{w}^{(t)})", "6")
    para(doc, "收敛判据取三条之一：梯度的无穷范数小于 tol、相邻两轮损失的相对变化小于 tol、"
              "达到最大迭代轮数。本文取 tol = 1e-10、最大轮数 30000，实测 1629 轮即收敛。")

    h3(doc, "2.1.4　步长的理论上界")
    para(doc, "梯度下降能否收敛、收敛多快，取决于目标函数的曲率。对逻辑回归，"
              "梯度是 Lipschitz 连续的，常数可由设计矩阵的最大特征值给出：")
    equation(doc, r"L_{\mathrm{Lip}} = \frac{\lambda_{\max}\left(\tilde{\mathbf{X}}^{\top}\tilde{\mathbf{X}}/N\right)}{4} + \lambda,\qquad \eta = \frac{1}{L_{\mathrm{Lip}}}", "7")
    para(doc, "取 η = 1/L_Lip 可以保证 J 单调下降。这个式子也解释了 2.1.6 节与 4.9 节的核心现象："
              "特征不做标准化时，X̃^T X̃/N 的最大特征值被最大量纲的特征（worst area，"
              "标准差约 568）直接放大，L_Lip 随之一飞冲天，可用的步长被压到 1e-6 量级，"
              "收敛慢到不可接受。本文实测标准化前后 L_Lip 相差约 12 万倍。")

    h3(doc, "2.1.5　数值稳定性处理")
    para(doc, "按定义直接实现会在两处出现数值问题。**第一处**是 sigmoid 中的 exp(-z)："
              "当 z 为大负数时 exp(-z) 上溢为 inf，1/(1+inf) 变成 0，虽然结果碰巧对，"
              "但会抛出警告；实现时按符号分段——z ≥ 0 用 1/(1+e^{-z})，"
              "z < 0 用 e^{z}/(1+e^{z})，两段的指数参数都非正，永远不会溢出。"
              "**第二处**是 ln p 与 ln(1-p) 的计算：p 可能因下溢变成 0，"
              "导致 ln 0 = -inf。利用恒等式")
    equation(doc, r"\ln\sigma(z) = -\ln\left(1 + e^{-z}\right),\qquad \ln\left(1-\sigma(z)\right) = -z - \ln\left(1 + e^{-z}\right)", "8")
    para(doc, "改用 np.logaddexp 一次算出，即使 z = -800 也能得到精确的 -800.0，"
              "而不是 -inf 或 0。这两条在 2.2.4 节的 log-sum-exp 归一化中还会用到。")

    h3(doc, "2.1.6　为什么必须引入 L2 正则")
    para(doc, "本数据在 30 维空间近似线性可分，此时**不带正则的极大似然估计根本不存在**。"
              "直观地说：只要把权重整体放大 k 倍，所有样本的 |z| 都会变大，"
              "每个样本的预测概率都更接近它自己的真实标签，训练损失单调下降；"
              "但 k → ∞ 时损失才趋于 0，也就是说最优权重在无穷远处，没有有限解。"
              "实验完全印证了这一点（表 10 与 notebook 输出）：λ = 0 时跑满 30000 轮仍未收敛，"
              "梯度停在 1e-4 量级，权重范数持续增长，牛顿法的步长也不断变大。")
    para(doc, "加入 L2 后，目标函数变成“拟合项 + λ‖w‖²”的严格凸函数，"
              "在“损失下降”与“权重变大”之间取得平衡，最优解唯一且有限。"
              "所以本文的 L2 不是为了追求更好的精度，而是为了**让极大似然估计有解**。"
              "λ 的取值在训练集内用 5 折交叉验证的对数损失选定（见 4.10 节），"
              "测试集不参与选择。")

    h3(doc, "2.1.7　程序流程图")
    para(doc, "图 1 给出逻辑回归训练的完整流程。图中步骤 1 的标准化只在训练集上估计均值与标准差；"
              "步骤 2 把偏置并入权重；步骤 4～6 是每一轮迭代的三件事——前向算概率、按定义算"
              "负对数似然、按解析式算梯度；步骤 7 更新参数；菱形判定给出三条出口"
              "（梯度足够小 / 损失不再变化 / 达到最大轮数），“否”分支回到步骤 4 继续迭代；"
              "步骤 8 输出最优参数与损失曲线。")
    figure(doc, FIG("fig1_lr_flowchart.png"), "图 1　逻辑回归（极大似然 + 梯度下降）训练流程图",
           width_cm=13.2)

    # -------------------------------------------------------------- 2.2
    h2(doc, "2.2　高斯朴素贝叶斯（Gaussian Naive Bayes）")

    h3(doc, "2.2.1　贝叶斯决策")
    para(doc, "生成式模型先对“数据是怎么产生的”建模，再用贝叶斯公式反推类别后验。"
              "对样本 x，后验为")
    equation(doc, r"P(y = c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y = c)\,P(y = c)}{P(\mathbf{x})}", "9")
    para(doc, "分母 P(x) 与类别无关，在比较各类别时相当于同一个常数，可以忽略。"
              "因此 MAP 决策只需比较分子 P(x|y=c)P(y=c)，即"
              "“类条件密度 × 类先验”。先验用训练集中各类样本占比估计，"
              "剩下的问题就是如何估计 30 维的类条件密度 P(x|y=c)。")

    h3(doc, "2.2.2　条件独立假设")
    para(doc, "直接估计 30 维联合密度需要极多的样本（维度灾难）。朴素贝叶斯引入一个强假设："
              "**在给定类别的条件下，各特征相互独立**。于是联合密度可以拆成一维密度的连乘：")
    equation(doc, r"P(\mathbf{x} \mid y = c) = \prod_{j=1}^{d} P(x_j \mid y = c)", "10")
    para(doc, "这把一个 30 维密度估计问题化简为 30 个一维密度估计问题，每个一维密度只需估计"
              "均值和方差两个参数，所需样本量从指数级降到线性级。代价是："
              "**当特征真的相关时，独立假设会把同一份信息重复计算多次**，"
              "导致后验概率被系统性高估。1.3 节已经测出本数据有 21 对 |r| ≥ 0.9 的特征，"
              "其中 mean radius 与 mean perimeter 的相关系数高达 0.9979，"
              "这不只是“轻微违背”，而是严重违背——这正是朴素贝叶斯在本数据上落后的根本原因。")

    h3(doc, "2.2.3　高斯类条件密度与参数估计")
    para(doc, "假设每个特征在每个类别下服从正态分布：")
    equation(doc, r"P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi\sigma_{cj}^{2}}}\exp\left(-\frac{(x_j - \mu_{cj})^{2}}{2\sigma_{cj}^{2}}\right)", "11")
    para(doc, "参数用极大似然闭式估计，即“组内均值”与“组内方差”：")
    equation(doc, r"\hat{\mu}_{cj} = \frac{1}{N_c}\sum_{i:\,y_i=c} x_{ij},\qquad \hat{\sigma}_{cj}^{2} = \frac{1}{N_c}\sum_{i:\,y_i=c}\left(x_{ij} - \hat{\mu}_{cj}\right)^{2}", "12")
    para(doc, "其中 N_c 是第 c 类的样本数。这里用除以 N_c 的极大似然方差（ddof = 0），"
              "与式 (12) 的推导一致；需要说明的是，这一选择与 scikit-learn 的 GaussianNB 相同"
              "（其内部用 np.var 的总体方差），只是有些教材写成除以 N_c-1 的无偏形式。"
              "在 N_c 约 150～250 的情况下两者差异在 0.5%% 以内，不影响任何结论。"
              "注意朴素贝叶斯**没有迭代过程**，参数一次闭式估计完成，训练只需扫描一遍数据。")

    h3(doc, "2.2.4　对数域计算与 log-sum-exp 归一化")
    para(doc, "把式 (10)(11)(12) 代回式 (9) 并取对数，连乘变成连加。"
              "忽略与类别无关的 ln P(x)，得到**未归一化的对数后验**：")
    equation(doc, r"s_c = \ln P(y=c) + \sum_{j=1}^{d}\left[-\frac{1}{2}\ln\left(2\pi\sigma_{cj}^{2}\right) - \frac{(x_j - \mu_{cj})^{2}}{2\sigma_{cj}^{2}}\right]", "13")
    para(doc, "需要输出概率时，对 {s_c} 做 log-sum-exp 归一化：")
    equation(doc, r"P(y = c \mid \mathbf{x}) = \frac{e^{s_c - \max_{c'} s_{c'}}}{\sum_{c'} e^{s_{c'} - \max_{c''} s_{c''}}}", "14")
    para(doc, "先减去最大值是纯数值技巧：分子分母同乘 e^{-max}，结果不变，"
              "却保证指数部分不超过 0，永远不会上溢。实现上还有一个细节："
              "把式 (13) 的平方展开成“x²/σ² + xμ/σ² + μ²/σ²”三部分后，"
              "整个 (N,2) 的对数后验可以写成两次矩阵乘法，"
              "不必对 60 个 (类别, 特征) 组合写 Python 循环，速度提升一个数量级。")

    h3(doc, "2.2.5　方差下限保护")
    para(doc, "若某个特征在某一类里取值恒定，则 σ²_cj = 0，式 (13) 依次出现"
              "ln 0 = -∞、除以 0、以及 0/0 = NaN 三种病态。任何一个 NaN 都会"
              "污染整条样本的后验概率，使预测彻底失效。解决办法是给方差设下限：")
    equation(doc, r"\tilde{\sigma}_{cj}^{2} = \max\left(\hat{\sigma}_{cj}^{2},\ \varepsilon\right),\qquad \varepsilon = 10^{-9}", "15")
    para(doc, "ε 取 1e-9 的理由：本数据最小类条件方差为 4.13e-6，比 ε 大三个数量级，"
              "因此这个下限在真实数据上**完全不改变任何估计值**（实测启用与关闭保护时"
              "后验概率的最大绝对差为 0）；而在退化情形下它能把 NaN 变成有限的数，"
              "让模型继续工作。3.5 节用“人为加入一列常数列”的对照实验把这个效果演示了出来。")


# ============================================================ 四、实验结果
def build_section4(doc):
    S = R["eval"]["standardization"]
    LC = R["eval"]["learning_curve"]
    TH = R["eval"]["threshold_table"]
    SEED = R["eval"]["seed_summary"]
    LAM = R["eval"]["lambda_table"]

    h1(doc, "四、实验结果")
    para(doc, "本节所有数字都由 notebook 直接输出（可用随代码附带的 results.json 复核）。"
              "测试集是固定的 171 条样本，其类别分布与训练集一致。")

    h2(doc, "4.1　实验设置与超参数")
    para(doc, "两个模型的超参数与确定方式见表 4。需要强调的是：λ 由训练集内部 5 折交叉验证"
              "选出来，测试集在超参数确定之前没有参与任何计算；"
              "朴素贝叶斯除了数值保护项外没有可调超参数。")
    table(doc,
          ["模型", "超参数", "取值", "确定方式"],
          [["逻辑回归", "L2 系数 λ", "%.2f" % LR["best_lambda"], "训练集内 5 折交叉验证对数损失最小"],
           ["逻辑回归", "学习率 η", "%.4f" % LR["eta"], "取梯度 Lipschitz 常数倒数 1/L"],
           ["逻辑回归", "最大迭代轮数", "30000", "留足余量，实测 %d 轮收敛" % LR["n_iter"]],
           ["逻辑回归", "收敛判据", "tol = 1e-10", "梯度范数或损失相对变化"],
           ["逻辑回归", "是否标准化", "是", "4.9 节证明必须标准化"],
           ["朴素贝叶斯", "方差下限 ε", "1e-9", "数值保护，真实数据上不影响结果"],
           ["朴素贝叶斯", "方差估计", "极大似然（除以 N_c）", "与原理推导一致"],
           ["两者共同", "训练/测试划分", "7 : 3 分层", "random_state = %d" % META["seed"]]],
          "表 4　模型超参数及其确定方式",
          col_widths=[2.4, 3.2, 3.6, 6.2], align_first_left=True, font_size=9.5)

    h2(doc, "4.2　测试集整体指标")
    para(doc, "表 5 给出主实验（seed = %d）下两种方法的全部指标。" % META["seed"])
    table(doc,
          ["模型", "准确率", "宏精确率", "宏召回率", "宏 F1", "宏 AP", "AUC", "ECE"],
          [["逻辑回归", f4(E_LR["acc"]), f4(E_LR["macro"]["precision"]), f4(E_LR["macro"]["recall"]),
            f4(E_LR["macro"]["f1"]), f4(E_LR["macro_ap"]), f4(E_LR["auc"]), f4(E_LR["ece"])],
           ["朴素贝叶斯", f4(E_NB["acc"]), f4(E_NB["macro"]["precision"]), f4(E_NB["macro"]["recall"]),
            f4(E_NB["macro"]["f1"]), f4(E_NB["macro_ap"]), f4(E_NB["auc"]), f4(E_NB["ece"])]],
          "表 5　两种方法在测试集上的性能对比（171 条样本，seed = 42）",
          col_widths=[2.8, 1.9, 1.9, 1.9, 1.7, 1.7, 1.7, 1.7], align_first_left=True, font_size=9.5)
    para(doc, "结论很清楚：**逻辑回归在全部 7 个指标上都优于朴素贝叶斯**。"
              "准确率高出 %.2f 个百分点（%s vs %s），宏 F1 高出 %.4f，宏平均 AP 高出 %.4f，"
              "AUC 高出 %.4f，ECE 也更低（校准更好）。值得注意的是，朴素贝叶斯的宏平均 AP "
              "达到 %.4f、AUC 达到 %.4f，说明它的**概率排序能力并不差**，"
              "主要差距体现在把概率压到 0 和 1 两端、导致阈值附近的判断失误上。"
              % (100 * (E_LR["acc"] - E_NB["acc"]), f4(E_LR["acc"]), f4(E_NB["acc"]),
                 E_LR["macro"]["f1"] - E_NB["macro"]["f1"],
                 E_LR["macro_ap"] - E_NB["macro_ap"], E_LR["auc"] - E_NB["auc"],
                 E_NB["macro_ap"], E_NB["auc"]))

    para(doc, "表 6 把指标拆到每个类别。两类测试样本量是 64 : 107，宏平均与加权平均差别不大，"
              "但能看出朴素贝叶斯的短板集中在**恶性类的召回率**：它的召回率只有 %.4f，"
              "即 %d 个恶性肿瘤里有 %d 个被漏诊；逻辑回归的召回率为 %.4f，只漏掉 %d 个。"
              "对医疗筛查而言，这个差距比总体准确率的差距更值得关注。"
              % (E_NB["per_class"]["0"]["recall"], E_NB["support"]["0"],
                 E_NB["per_class"]["0"]["fn"],
                 E_LR["per_class"]["0"]["recall"], E_LR["per_class"]["0"]["fn"]))
    table(doc,
          ["模型", "类别", "精确率", "召回率", "F1", "AP", "样本数"],
          [["逻辑回归", "恶性（0）",
            f4(E_LR["per_class"]["0"]["precision"]), f4(E_LR["per_class"]["0"]["recall"]),
            f4(E_LR["per_class"]["0"]["f1"]), f4(E_LR["ap_per_class"]["0"]), str(E_LR["support"]["0"])],
           ["逻辑回归", "良性（1）",
            f4(E_LR["per_class"]["1"]["precision"]), f4(E_LR["per_class"]["1"]["recall"]),
            f4(E_LR["per_class"]["1"]["f1"]), f4(E_LR["ap_per_class"]["1"]), str(E_LR["support"]["1"])],
           ["逻辑回归", "宏平均",
            f4(E_LR["macro"]["precision"]), f4(E_LR["macro"]["recall"]),
            f4(E_LR["macro"]["f1"]), f4(E_LR["macro_ap"]), "171"],
           ["朴素贝叶斯", "恶性（0）",
            f4(E_NB["per_class"]["0"]["precision"]), f4(E_NB["per_class"]["0"]["recall"]),
            f4(E_NB["per_class"]["0"]["f1"]), f4(E_NB["ap_per_class"]["0"]), str(E_NB["support"]["0"])],
           ["朴素贝叶斯", "良性（1）",
            f4(E_NB["per_class"]["1"]["precision"]), f4(E_NB["per_class"]["1"]["recall"]),
            f4(E_NB["per_class"]["1"]["f1"]), f4(E_NB["ap_per_class"]["1"]), str(E_NB["support"]["1"])],
           ["朴素贝叶斯", "宏平均",
            f4(E_NB["macro"]["precision"]), f4(E_NB["macro"]["recall"]),
            f4(E_NB["macro"]["f1"]), f4(E_NB["macro_ap"]), "171"]],
          "表 6　逐类别（one-vs-rest）精确率、召回率、F1 与 AP",
          col_widths=[2.6, 2.4, 2.2, 2.2, 2.0, 2.0, 1.8], align_first_left=True, font_size=9.5)

    h2(doc, "4.3　混淆矩阵")
    err_lr = int(E_LR["cm"][0][1] + E_LR["cm"][1][0])
    err_nb = int(E_NB["cm"][0][1] + E_NB["cm"][1][0])
    para(doc, "图 3 直观展示两种方法的错误分布。逻辑回归只错 %d 条（漏诊 %d 条、误诊 %d 条），"
              "朴素贝叶斯错 %d 条（漏诊 %d 条、误诊 %d 条），错误量是前者的 %.1f 倍，"
              "且主要错在把恶性判成良性这个最危险的方向上。"
              % (err_lr, E_LR["per_class"]["0"]["fn"], E_LR["per_class"]["0"]["fp"],
                 err_nb, E_NB["per_class"]["0"]["fn"], E_NB["per_class"]["0"]["fp"],
                 err_nb / max(1, err_lr)))
    figure(doc, FIG("fig_confusion.png"), "图 3　测试集混淆矩阵对比（左：逻辑回归；右：朴素贝叶斯）",
           width_cm=13.6)

    h2(doc, "4.4　P-R 曲线与 AP")
    para(doc, "图 4 是两个类别的 P-R 曲线。逻辑回归的曲线在两个类别上都完全包住朴素贝叶斯，"
              "这正是宏平均 AP 更高的几何含义：在任意召回率水平上，逻辑回归的精确率都不低于朴素贝叶斯。"
              "从数值上看，恶性类的 AP 差距（%.4f vs %.4f）明显大于良性类（%.4f vs %.4f），"
              "说明朴素贝叶斯在少数类上的退化更严重——它的参数按类别各自估计，"
              "少数类样本更少，估计方差更大。"
              % (E_LR["ap_per_class"]["0"], E_NB["ap_per_class"]["0"],
                 E_LR["ap_per_class"]["1"], E_NB["ap_per_class"]["1"]))
    figure(doc, FIG("fig_pr_curves.png"), "图 4　两种方法的 P-R 曲线（左：恶性类；右：良性类）",
           width_cm=14.6)

    h2(doc, "4.5　阈值对精确率与召回率的影响")
    th_lr, th_nb = TH["逻辑回归"], TH["朴素贝叶斯"]
    idx_hi = next(i for i, x in enumerate(th_lr) if x["recall"] >= 0.999)
    para(doc, "图 5 给出阈值从 0.05 扫到 0.95 时精确率、召回率与 F1 的变化。两个模型的曲线形状差别很大："
              "逻辑回归在阈值 %.2f 处就能把召回率做到 %.4f（此时精确率仍有 %.4f）；"
              "而朴素贝叶斯即使把阈值压到 0.05，召回率也只有 %.4f，"
              "且阈值在 0.20～0.55 之间时曲线完全平坦。原因在于两者的概率分布不同："
              "逻辑回归的预测概率连续铺开在 0～1 之间，阈值可以平滑调节；"
              "朴素贝叶斯的概率大量堆在两端，约 94.7%% 的样本置信度高于 0.95，"
              "阈值落在中间区间时几乎没有样本被移动。"
              % (th_lr[idx_hi]["threshold"], th_lr[idx_hi]["recall"], th_lr[idx_hi]["precision"],
                 th_nb[0]["recall"]))
    figure(doc, FIG("fig_threshold.png"), "图 5　阈值对精确率、召回率与 F1 的影响（正类为恶性）",
           width_cm=14.6)
    para(doc, "按 F1 最大准则，逻辑回归的最优阈值是 %.2f（F1 = %.4f），朴素贝叶斯是 %.2f"
              "（F1 = %.4f）。如果目标是尽可能不漏诊，逻辑回归可以把阈值降到 %.2f 使召回率达到 1.0，"
              "代价是精确率降到 %.4f；朴素贝叶斯做不出这样的取舍，因为它的概率已经饱和。"
              "这也说明：**概率模型的价值不仅在于排序，还在于概率本身是否可信**。"
              % (R["eval"]["best_threshold"]["逻辑回归"]["threshold"],
                 R["eval"]["best_threshold"]["逻辑回归"]["f1"],
                 R["eval"]["best_threshold"]["朴素贝叶斯"]["threshold"],
                 R["eval"]["best_threshold"]["朴素贝叶斯"]["f1"],
                 th_lr[idx_hi]["threshold"], th_lr[idx_hi]["precision"]))

    h2(doc, "4.6　概率校准")
    para(doc, "图 6 左边是可靠性曲线：横轴为平均置信度，纵轴为实际正确率，对角线代表完美校准。"
              "逻辑回归的曲线贴近对角线，ECE = %.4f；朴素贝叶斯的曲线整体偏离对角线，ECE = %.4f。"
              "右边是预测概率直方图：朴素贝叶斯有 94.7%% 的样本概率落在 0.05 以下或 0.95 以上，"
              "逻辑回归只有 74.3%%。这正是式 (10) 的条件独立假设被违背后的典型症状——"
              "把 30 个高度相关的特征当成独立证据相乘，等于把同一份证据数了好几遍，"
              "后验概率因此被系统性放大。"
              % (E_LR["ece"], E_NB["ece"]))
    figure(doc, FIG("fig_calibration.png"), "图 6　概率校准分析（左：可靠性曲线；右：预测概率分布）",
           width_cm=14.6)

    # ---------------------------------------------------------- 4.7 稳定性
    h2(doc, "4.7　十种随机种子的稳定性")
    para(doc, "单次划分只有 171 条测试样本，错一条就差 0.58 个百分点，因此一次结果不足以支撑结论。"
              "用 10 个不同的随机种子各做一次完整的划分—训练—评估，结果见表 7 与图 7。")
    rows7 = []
    for name in ["逻辑回归", "朴素贝叶斯"]:
        s = SEED[name]
        rows7.append([name,
                      "%.4f±%.4f" % (s["acc"][0], s["acc"][1]),
                      "%.4f±%.4f" % (s["macro_P"][0], s["macro_P"][1]),
                      "%.4f±%.4f" % (s["macro_F1"][0], s["macro_F1"][1]),
                      "%.4f±%.4f" % (s["macro_AP"][0], s["macro_AP"][1])])
    table(doc,
          ["模型", "准确率", "宏精确率", "宏 F1", "宏 AP"],
          rows7,
          "表 7　10 个随机种子下的指标（均值 ± 样本标准差）",
          col_widths=[2.6, 3.3, 3.1, 3.1, 3.3], align_first_left=True, font_size=9.0)
    para(doc, "10 个种子上逻辑回归的宏 F1 为 %.4f ± %.4f，朴素贝叶斯为 %.4f ± %.4f，"
              "前者在**每一个种子**上都更高；宏 AP 的差距（%.4f vs %.4f）同样稳定。"
              "两个模型的标准差都在 0.01～0.02 量级，说明单次实验的波动确实不小，"
              "而两者之间 0.04 左右的宏 F1 差距明显大于这个波动，"
              "因此“逻辑回归更好”这个结论不是某一次划分的偶然。"
              % (SEED["逻辑回归"]["macro_F1"][0], SEED["逻辑回归"]["macro_F1"][1],
                 SEED["朴素贝叶斯"]["macro_F1"][0], SEED["朴素贝叶斯"]["macro_F1"][1],
                 SEED["逻辑回归"]["macro_AP"][0], SEED["朴素贝叶斯"]["macro_AP"][0]))
    figure(doc, FIG("fig_seed_stability.png"), "图 7　10 个随机种子下两种方法的指标分布",
           width_cm=14.2)

    # ---------------------------------------------------------- 4.8 学习曲线
    h2(doc, "4.8　训练集规模的影响（学习曲线）")
    para(doc, "从偏差—方差的角度看，朴素贝叶斯参数少、方差低，通常在小样本下更有优势。"
              "为检验这一点，把训练集按类别分层抽取 10%%～100%%，测试集保持不变（表 8、图 8）。"
              "结果与直觉相反：**样本最少时朴素贝叶斯也没能反超**，此时两者的差距最小"
              "（%d 条训练样本时宏 F1 为 %.4f 对 %.4f），而样本增加后差距稳定扩大到 0.05～0.06"
              "（70%% 时最大，为 %.4f）。"
              % (LC[0]["n_train"], LC[0]["LR_f1"], LC[0]["GNB_f1"],
                 max(r["LR_f1"] - r["GNB_f1"] for r in LC)))
    table(doc,
          ["训练比例", "训练样本数", "逻辑回归 宏 F1", "朴素贝叶斯 宏 F1", "差距"],
          [["%.0f%%" % (100 * r["frac"]), str(r["n_train"]), f4(r["LR_f1"]), f4(r["GNB_f1"]),
            "%+.4f" % (r["LR_f1"] - r["GNB_f1"])] for r in LC],
          "表 8　训练集规模对测试集宏 F1 的影响（测试集固定）",
          col_widths=[3.0, 3.4, 3.4, 3.4, 2.4], font_size=9.5)
    para(doc, "这条学习曲线说明了三点。第一，朴素贝叶斯的宏 F1 从 40 条样本起就基本落在 "
              "0.92～0.94 的窄带里，增加样本几乎没有改善——它的瓶颈是**假设偏差**，不是样本量；"
              "第二，逻辑回归的性能总体随样本数上升（20%%→30%% 处有一次小幅回落，从 %.4f 到 %.4f，"
              "属于单一划分带来的波动），说明它的瓶颈是方差，数据越多越能学出可靠的判别方向；"
              "第三，本题的特征多达 30 个且近似线性可分，"
              "判别式模型能有效利用样本，生成式模型的独立假设反而成了硬约束。"
              "这也提醒我们：**“小样本用朴素贝叶斯”是有前提的经验，不是普遍规律**。"
              % (LC[1]["LR_f1"], LC[2]["LR_f1"]))
    figure(doc, FIG("fig_learning_curve.png"), "图 8　学习曲线：训练集规模对两种方法的影响",
           width_cm=12.8)

    # ---------------------------------------------------------- 4.9 标准化消融
    h2(doc, "4.9　特征标准化对梯度下降的影响")
    para(doc, "表 9 对比了同一模型在原始尺度与标准化数据上的数值条件。两者的差别是数量级的："
              "梯度 Lipschitz 常数相差 %.0f 倍，可用的步长相差 %.0f 倍，"
              "设计矩阵的条件数相差 %.0e 倍。结果是，在各自取理论上界步长、同样跑 30000 轮的前提下，"
              "标准化版本 1629 轮就收敛到 %.6f 的损失，测试准确率 %.4f；"
              "原始尺度版本 30000 轮后仍未收敛，损失停在 %.4f，准确率只有 %.4f。"
              % (S["L_raw"] / S["L_std"], S["eta_ratio"], S["cond_ratio"],
                 S["loss_std"], S["acc_std"], S["loss_raw"], S["acc_raw"]))
    table(doc,
          ["指标", "标准化后", "原始尺度", "倍数"],
          [["梯度 Lipschitz 常数 L", "%.4f" % S["L_std"], "%.1f" % S["L_raw"],
            "%.0f" % (S["L_raw"] / S["L_std"])],
           ["理论上界步长 η = 1/L", "%.4f" % S["eta_std"], "%.2e" % S["eta_raw"],
            "%.0f" % S["eta_ratio"]],
           ["X̃ᵀX̃/N 的条件数", "%.2e" % S["cond_std"], "%.2e" % S["cond_raw"],
            "%.0e" % S["cond_ratio"]],
           ["30000 轮后训练损失", "%.6f" % S["loss_std"], "%.6f" % S["loss_raw"], "—"],
           ["是否收敛", "是" if S["conv_std"] else "否", "是" if S["conv_raw"] else "否", "—"],
           ["测试集准确率", "%.4f" % S["acc_std"], "%.4f" % S["acc_raw"], "—"]],
          "表 9　特征标准化前后的数值条件与收敛行为（λ = 1e-2）",
          col_widths=[5.0, 3.4, 3.6, 3.0], align_first_left=True, font_size=9.5)
    para(doc, "图 9 左边是两条损失曲线（各自按 1/L 取步长），右边是梯度范数的下降速度（对数轴）。"
              "未标准化的曲线在前 30000 轮里几乎是一条水平线，梯度范数只降了一个数量级；"
              "标准化后则很快压到 1e-6 以下。这为“逻辑回归必须先标准化”提供了直接的实验证据，"
              "背后的道理就是式 (7)：步长被最大量纲的特征牢牢限制住。")
    figure(doc, FIG("fig_standardization.png"), "图 9　特征标准化对梯度下降收敛速度的影响",
           width_cm=14.6)

    # ---------------------------------------------------------- 4.10 正则系数
    h2(doc, "4.10　L2 系数 λ 的选择与敏感性")
    para(doc, "表 10 是 λ 网格上的完整结果，图 10 把交叉验证损失与权重范数画在一起。"
              "λ 从 0 增大到 1，权重 L2 范数从 %.2f 单调压到 %.2f；"
              "交叉验证对数损失先降后升，在 λ = %.2f 处最小（%.5f）。"
              "λ = 0 与 λ = 1e-4 都是跑满 30000 轮未收敛（stop 一列为 max_iter），"
              "这正是 2.1.6 节“近似线性可分时 MLE 不存在”的直接证据。"
              % (LAM[0]["w_norm"], LAM[-1]["w_norm"], LR["best_lambda"],
                 min(x["CV_logloss"] for x in LAM)))
    table(doc,
          ["λ", "CV 对数损失", "迭代轮数", "停止原因", "训练损失", "测试准确率", "权重范数"],
          [["%g" % x["lambda"], "%.5f" % x["CV_logloss"], str(x["GD_iters"]), x["stop"],
            "%.6f" % x["train_loss"], "%.4f" % x["test_acc"], "%.4f" % x["w_norm"]] for x in LAM],
          "表 10　L2 系数 λ 对收敛性、交叉验证损失与性能的影响",
          col_widths=[1.8, 2.6, 2.2, 2.6, 2.4, 2.2, 2.2], font_size=9.5)
    para(doc, "λ 的选择体现了典型的偏差—方差权衡：λ 太小则权重过大、收敛慢、逼近不可分的极端解，"
              "测试准确率在 λ = 1e-3 时反而是最高的 0.9883；λ 太大则正则项压过拟合项，"
              "训练损失从 0.096 升到 0.378，测试准确率跌到 0.9415，属于欠拟合。"
              "本文按预先设定的规则“取 CV 对数损失最小”选了 λ = 0.01，"
              "而不是事后按测试集挑一个最好看的数字——后者会让评估失去意义。")
    figure(doc, FIG("fig_lambda_grid.png"), "图 10　L2 系数 λ 对交叉验证损失与权重范数的影响",
           width_cm=13.2)

    # ---------------------------------------------------------- 4.11 代价
    h2(doc, "4.11　计算代价")
    para(doc, "表 11 是同一台机器上单线程重复 5 次的训练耗时。逻辑回归需要 %d 轮迭代，"
              "平均 %.1f ms；高斯朴素贝叶斯是闭式估计，平均 %.2f ms，**快约 %.0f 倍**。"
              "不过要注意，这个差距在本题规模下没有实际意义——两者都在毫秒级。"
              "逻辑回归的代价与迭代轮数成正比，而迭代轮数又取决于条件数，"
              "所以“数据有没有标准化”直接决定了它的训练开销，这与 4.9 节的结论一致。"
              % (LR["n_iter"], LR["train_time_ms"], NB_["train_time_ms"],
                 LR["train_time_ms"] / NB_["train_time_ms"]))
    table(doc,
          ["模型", "平均耗时 / ms", "标准差 / ms", "迭代轮数", "复杂度"],
          [["逻辑回归", "%.2f" % LR["train_time_ms"], "%.2f" % LR["train_time_std_ms"],
            str(LR["n_iter"]), "O(T·N·d)"],
           ["朴素贝叶斯", "%.2f" % NB_["train_time_ms"], "%.2f" % NB_["train_time_std_ms"],
            "0（闭式）", "O(N·d)"]],
          "表 11　训练耗时对比（重复 5 次取平均，单线程）",
          col_widths=[3.2, 3.2, 3.0, 2.6, 3.0], align_first_left=True, font_size=9.5)

    # ---------------------------------------------------------- 4.12 特征权重
    h2(doc, "4.12　模型参数与特征解释")
    para(doc, "逻辑回归的权重可以直接比较（特征已标准化，权重绝对值表示该特征对判别的贡献）。"
              "权重为正表示把样本推向良性、为负表示推向恶性。表 12 列出绝对值最大的 10 个特征，"
              "图 11 是所有特征中最大的 12 个的条形图。")
    table(doc,
          ["特征", "逻辑回归权重", "恶性组均值", "良性组均值"],
          [["%s" % c["name"], "%.4f" % c["weight"], "%.4f" % c["mean0"], "%.4f" % c["mean1"]]
           for c in R["eval"]["coef_top"]],
          "表 12　逻辑回归权重绝对值最大的 10 个特征（均已标准化）",
          col_widths=[5.4, 3.6, 3.4, 3.4], align_first_left=True, font_size=9.5)
    para(doc, "解读这些权重要小心两点。第一，%d 个权重里只有 %d 个为正，"
              "说明绝大多数形态指标都是“数值越大越像恶性”，符合医学常识："
              "恶性肿瘤的细胞核更大、形状更不规则、凹陷更明显。"
              "第二，权重是在**其余 29 个特征同时存在**的条件下估计的，"
              "由于 worst radius / worst area / worst perimeter 之间相关系数超过 0.97，"
              "它们的权重会相互“分摊”，单独拿某个权重说“这个指标最重要”是不严谨的；"
              "要判断特征重要性，更稳妥的做法是做消融或者看置换重要性。"
              % (R["eval"]["coef_pos"] + R["eval"]["coef_neg"], R["eval"]["coef_pos"]))
    figure(doc, FIG("fig_lr_coef.png"), "图 11　逻辑回归权重绝对值最大的 12 个特征",
           width_cm=13.0)


# ============================================================ 五、结论与分析
def build_section5(doc):
    h1(doc, "五、结论与分析")

    h2(doc, "5.1　结论")
    para(doc, "在乳腺癌二分类任务上，**手写逻辑回归在所有评价指标上都优于手写高斯朴素贝叶斯**。"
              "固定划分（seed = 42）下，逻辑回归的宏平均 AP 为 %.4f、宏平均 F1 为 %.4f、"
              "准确率为 %.4f；朴素贝叶斯分别为 %.4f、%.4f、%.4f。"
              "换 10 个随机种子后结论不变：逻辑回归宏 F1 为 %.4f ± %.4f，"
              "朴素贝叶斯为 %.4f ± %.4f，且逻辑回归在每一个种子上都更高。"
              "差距的主要来源不是实现质量（两者的实现都已用牛顿法、sklearn.metrics 和理论性质"
              "三方面验证过），而是**两种方法对数据的假设与真实数据之间的吻合程度不同**。"
              % (E_LR["macro_ap"], E_LR["macro"]["f1"], E_LR["acc"],
                 E_NB["macro_ap"], E_NB["macro"]["f1"], E_NB["acc"],
                 R["eval"]["seed_summary"]["逻辑回归"]["macro_F1"][0],
                 R["eval"]["seed_summary"]["逻辑回归"]["macro_F1"][1],
                 R["eval"]["seed_summary"]["朴素贝叶斯"]["macro_F1"][0],
                 R["eval"]["seed_summary"]["朴素贝叶斯"]["macro_F1"][1]))
    para(doc, "如果只允许一句话总结：**逻辑回归赢在它不对特征之间的关系做假设，"
              "朴素贝叶斯输在它的独立性假设在这份数据上严重不成立。**")

    h2(doc, "5.2　逻辑回归更好的原因")
    bullet(doc, "（1）", "**条件独立假设在本数据上严重违背。** 30 个特征共有 %d 对两两组合，"
                         "其中 |r| ≥ 0.9 的有 %d 对，mean radius 与 mean perimeter 的相关系数高达 0.9979。"
                         "朴素贝叶斯把 30 个特征当作独立证据连乘，等于把同一份形态信息数了好几遍，"
                         "先验与似然的相对权重被扭曲。这直接体现在概率层面：朴素贝叶斯的 ECE 为 %.4f，"
                         "逻辑回归为 %.4f；朴素贝叶斯有 94.7%% 的样本预测概率落在两端的 5%% 区间内，"
                         "逻辑回归只有 74.3%%。过度自信的概率让阈值调节失去意义（见 4.5 节）。"
           % (30 * 29 // 2, META["n_strong_corr_pairs_0.9"], E_NB["ece"], E_LR["ece"]))
    bullet(doc, "（2）", "**判别式模型与生成式模型的优化目标不同。** 逻辑回归直接优化条件似然 "
                         "P(y|x)，优化的正是分类任务本身关心的量；朴素贝叶斯优化联合似然 "
                         "P(x,y)，分类只是它的副产物。当类条件密度的假设（高斯、独立）不成立时，"
                         "联合似然的最优解与分类错误率的最小解会产生偏差——这是判别式模型在"
                         "假设不完美时更稳健的经典理由。本数据中，朴素贝叶斯的宏平均 AP 仍有 %.4f，"
                         "说明它学到的排序大体是对的，但概率尺度失真，"
                         "导致 0.5 阈值附近的判断（精确率 %.4f、召回率 %.4f）明显变差。"
           % (E_NB["macro_ap"], E_NB["per_class"]["0"]["precision"], E_NB["per_class"]["0"]["recall"]))
    bullet(doc, "（3）", "**偏差—方差：本数据的瓶颈是偏差，不是方差。** 高斯朴素贝叶斯用对角协方差"
                         "近似真实的协方差矩阵，这是结构性偏差，样本再多也消不掉；它的参数估计方差确实小。"
                         "逻辑回归偏差小、参数多（30 个权重 + 1 个截距），方差相对大。"
                         "而 398 条训练样本支撑 31 个参数并不吃力（样本特征比约 13 : 1），"
                         "所以逻辑回归的方差没有成为负担。学习曲线（4.8 节）是这一点的直接证据："
                         "朴素贝叶斯的宏 F1 从 40 条样本起就停在 0.92～0.94 的平台，"
                         "增加样本毫无改善，说明它已经到达偏差决定的上限；逻辑回归则总体随样本增加而改善。")
    bullet(doc, "（4）", "**概率校准的质量影响实际可用性。** 逻辑回归的损失函数就是对数似然，"
                         "输出的概率天然接近真实频率；朴素贝叶斯的概率来自密度比，"
                         "在独立假设违背时会被系统性放大。在医疗场景里这一点尤其重要："
                         "医生要根据概率决定是否穿刺活检，一个预测为 0.99 良性的结果如果实际只有 "
                         "0.90 的把握，代价就是漏诊。4.6 节的可靠性曲线把这个问题量化了出来。")
    bullet(doc, "（5）", "**计算代价不构成选择理由。** 朴素贝叶斯确实快得多（%.2f ms 对 %.2f ms，"
                         "约 %.0f 倍），但两者都在毫秒级，而模型精度差 4～5 个百分点。"
                         "只有当数据规模大到逻辑回归需要小时级训练时，速度才会成为决定性因素。"
           % (NB_["train_time_ms"], LR["train_time_ms"],
              LR["train_time_ms"] / NB_["train_time_ms"]))

    h2(doc, "5.3　朴素贝叶斯的优势与适用场景")
    para(doc, "这不意味着朴素贝叶斯没有价值。它的优点在别的场景里非常突出："
              "训练是闭式估计、只需扫描一遍数据，可以**增量更新**（来一批新样本只改均值和方差）；"
              "没有需要调的迭代超参数；参数少所以方差低，在**样本极少**时不容易过拟合；"
              "每个特征独立贡献一个对数项，可解释性强。最典型的成功场景是**高维稀疏的文本分类**"
              "（如垃圾邮件识别）：词袋特征维度上万、样本几千，特征之间的相关性远小于本题的形态指标，"
              "独立假设近似成立，此时朴素贝叶斯的快速与稳健非常划算。"
              "本题之所以落败，是因为三个条件同时不满足：特征强相关、维度不算高、样本量足够。")

    h2(doc, "5.4　实验的局限与可改进之处")
    bullet(doc, "（1）", "**单一数据集。** 结论只在乳腺癌数据上成立，换一个特征相关性弱的数据集"
                         "（例如各特征独立的高斯混合数据）朴素贝叶斯很可能反超。"
                         "更严谨的做法是在多个数据集上重复同一套实验。")
    bullet(doc, "（2）", "**未直接检验“独立假设是主因”这一诊断。** 一个可验证的改进是："
                         "把高斯朴素贝叶斯换成**二次判别分析（QDA）**，即允许每个类别有自己的完整"
                         "协方差矩阵。如果 QDA 的表现接近逻辑回归，就说明性能差距确实来自忽略特征相关；"
                         "如果还差得远，则说明问题出在训练样本不足导致协方差估计不准。")
    bullet(doc, "（3）", "**未做特征选择或降维。** 30 个特征里有大量冗余（21 对 |r| ≥ 0.9），"
                         "主成分分析后保留若干主成分，朴素贝叶斯的独立假设会更容易满足，"
                         "这是提升它表现的可行方向。")
    bullet(doc, "（4）", "**未做代价敏感学习。** 医疗场景里漏诊与误诊的代价不同，"
                         "应把两类错误的代价比写进目标函数（加权交叉熵），而不是只在阈值上做补偿。")
    bullet(doc, "（5）", "**未做统计显著性检验。** 两次实验的差异是否显著，"
                         "严格来说需要 McNemar 检验或重复交叉验证的配对 t 检验。"
                         "本文用 10 个种子给出均值与标准差作为替代，能说明差距稳定但不够严格。")
    para(doc, "**总体认识**：这次实验最重要的收获不是“逻辑回归赢了”，而是理解了两者的胜负"
              "由什么决定——**模型假设与数据结构的匹配程度**。逻辑回归用 31 个参数换来了"
              "不做独立性假设的自由；朴素贝叶斯用独立性假设换来了极少的方差和极快的训练。"
              "在选择模型之前，先看清数据有没有强相关、样本够不够，"
              "比背下“哪个算法更准”的结论重要得多。")


# ============================================================ 六、参考资料
def build_section6(doc):
    h1(doc, "六、参考资料")
    refs = [
        "周志华. 机器学习[M]. 北京: 清华大学出版社, 2016: 第 3 章（线性模型）、第 7 章（贝叶斯分类器）.",
        "李航. 统计学习方法（第 2 版）[M]. 北京: 清华大学出版社, 2019: "
        "第 4 章（朴素贝叶斯法）、第 6 章（逻辑斯谛回归与最大熵模型）.",
        "Bishop C M. Pattern Recognition and Machine Learning[M]. New York: Springer, 2006: "
        "Chapter 4 (Linear Models for Classification), Section 4.2 (Probabilistic Generative Models).",
        "Boyd S, Vandenberghe L. Convex Optimization[M]. Cambridge: Cambridge University Press, 2004: "
        "Section 9.2 (Steepest Descent), Section 9.3 (Gradient Descent Method).",
        "Pedregosa F, et al. Scikit-learn: Machine Learning in Python[J]. "
        "Journal of Machine Learning Research, 2011, 12: 2825-2830.",
        "scikit-learn developers. Toy datasets: load_breast_cancer[EB/OL]. "
        "https://scikit-learn.org/stable/datasets/toy_dataset.html",
        "scikit-learn developers. Metrics and scoring（含 average_precision_score）[EB/OL]. "
        "https://scikit-learn.org/stable/modules/model_evaluation.html",
        "scikit-learn developers. Probability calibration[EB/OL]. "
        "https://scikit-learn.org/stable/modules/calibration.html",
        "Harris C R, et al. Array programming with NumPy[J]. Nature, 2020, 585: 357-362.",
        "Hunter J D. Matplotlib: A 2D Graphics Environment[J]. "
        "Computing in Science & Engineering, 2007, 9(3): 90-95.",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Pt(24)
        pf.first_line_indent = Pt(-24)
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.space_after = Pt(2)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run("[%d] %s" % (i, r))
        set_run_font(run, BODY_FONT, 10.5)

    h2(doc, "附：AI 工具使用声明")
    para(doc, "按作业模板要求，如实说明 AI 工具的使用情况：本作业 notebook 的代码框架、"
              "部分注释文本、实验脚本的组织方式，以及 Word 报告的排版代码，"
              "在 AI 编程助手（DeepSeek 系列模型）的辅助下完成；"
              "算法原理的推导与核对、实验方案的设计、结果的解读与结论的得出由本人完成，"
              "并对每一项数字做了交叉验证——手写评估指标与 sklearn.metrics 逐项比对（最大差 1.11e-16）、"
              "梯度下降的解与牛顿法比对（最大差 8.15e-4）、"
              "所有表格与图片均由随附 notebook 重跑生成。"
              "报告中引用的全部数字都可以通过重跑 notebook 复现（随机种子已固定）。")


# ============================================================ 主程序
def main():
    doc = setup_document()
    cp = doc.core_properties
    cp.author = "XXX"
    cp.last_modified_by = "XXX"
    cp.title = "机器学习课程作业 2（B 组）报告"
    build_cover(doc)
    build_section1(doc)
    build_section2(doc)
    build_section2b(doc)
    build_section3(doc)
    build_section4(doc)
    build_section5(doc)
    build_section6(doc)
    doc.save(OUT)
    print("已生成:", OUT)


if __name__ == "__main__":
    main()
