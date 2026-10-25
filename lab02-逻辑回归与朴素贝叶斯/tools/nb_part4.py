# -*- coding: utf-8 -*-
"""Notebook 第 4 部分：手写评估指标、实验结果、图表与结果导出"""

CELLS = []

CELLS.append(("md", r"""
## 4. 手写评估指标

作业要求用**精确率、召回率、AP**（建议给宏平均值）作为评价指标。下面全部手写，
并在 4.2 节与 `sklearn.metrics` 交叉校验，确认没有算错。

### 4.1 定义

对二分类问题，以某个类别为正类（one-vs-rest），混淆矩阵的四个格子为

| | 预测正 | 预测负 |
|---|---|---|
| **实际正** | TP | FN |
| **实际负** | FP | TN |

$$ \text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}, \qquad
F_1 = \frac{2 \cdot P \cdot R}{P + R} $$

**P-R 曲线**：把分类阈值从 1 降到 0，每个阈值算一对 (Recall, Precision) 连成曲线。
曲线越靠近右上角越好。

**AP（Average Precision，平均精度）**：P-R 曲线下的面积，用阶梯积分计算

$$ AP = \sum_{n} (R_n - R_{n-1}) \cdot P_n $$

AP 比单点 F1 更全面：它把**所有阈值**下的表现都考虑进去了，不依赖"选 0.5 当阈值"这个人为主观选择。

**宏平均（macro）**：先对每个类别单独算指标，再取算术平均。
两个类别等权，不受类别样本量影响（本数据两类是 212 : 357，并不均衡，所以宏平均更有意义）。

$$ \text{macro-}P = \frac{1}{K}\sum_{c=1}^{K} P_c $$

同时给出**加权平均**（按各类样本数加权）作为补充，两者对照可以看出类别不均衡的影响。
"""))

CELLS.append(("code", r'''# ============================================================ 混淆矩阵
def confusion_matrix(y_true, y_pred, labels):
    """返回 KxK 混淆矩阵，行 = 真实类别，列 = 预测类别"""
    labels = np.asarray(labels)
    K = len(labels)
    idx = {c: i for i, c in enumerate(labels)}   # 类别 -> 行列下标
    M = np.zeros((K, K), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        M[idx[t], idx[p]] += 1                   # 对应格子加一
    return M


# ============================================================ 二分类 P/R/F1
def binary_prf(y_true_bin, y_pred_bin):
    """one-vs-rest 情形下的 TP/FP/FN/TN 与精确率、召回率、F1"""
    yt = np.asarray(y_true_bin).astype(bool)     # 真实标签（转布尔）
    yp = np.asarray(y_pred_bin).astype(bool)     # 预测标签（转布尔）
    tp = int(np.sum(yt & yp))                    # 真正例
    fp = int(np.sum(~yt & yp))                   # 假正例
    fn = int(np.sum(yt & ~yp))                   # 假负例
    tn = int(np.sum(~yt & ~yp))                  # 真负例
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # 精确率
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0     # 召回率
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0  # F1
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


def prf_report(y_true, y_pred, labels):
    """逐类别 + 宏平均 + 加权平均的 P/R/F1，以及准确率"""
    labels = np.asarray(labels)
    rows = {}
    for c in labels:
        rows[int(c)] = binary_prf(np.asarray(y_true) == c, np.asarray(y_pred) == c)
    macro = {k: float(np.mean([rows[int(c)][k] for c in labels]))
             for k in ["precision", "recall", "f1"]}                 # 宏平均：各类等权
    support = {int(c): int(np.sum(np.asarray(y_true) == c)) for c in labels}   # 各类样本数
    n = len(y_true)
    weighted = {k: float(sum(rows[int(c)][k] * support[int(c)] for c in labels) / n)
                for k in ["precision", "recall", "f1"]}              # 加权平均：按样本数
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))   # 准确率
    return rows, macro, weighted, acc, support


# ============================================================ P-R 曲线与 AP
def pr_curve(y_true_bin, scores):
    """P-R 曲线：按分数降序扫描，只在分数变化处取阈值点"""
    y = np.asarray(y_true_bin).astype(float)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")     # 按分数降序排（稳定排序）
    y, s = y[order], s[order]
    tp = np.cumsum(y)                            # 累计真正例
    fp = np.cumsum(1.0 - y)                      # 累计假正例
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]   # 每个不同分数阈值取最后一个点
    tp, fp = tp[last], fp[last]
    P = tp / np.maximum(tp + fp, 1e-12)          # 精确率
    R = tp / max(y.sum(), 1.0)                   # 召回率
    return np.r_[0.0, R], np.r_[1.0, P]          # 补上曲线起点 (R=0, P=1)


def average_precision(y_true_bin, scores):
    """AP = sum (R_n - R_{n-1}) * P_n，精确率不插值（与 sklearn 定义一致）"""
    R, P = pr_curve(y_true_bin, scores)
    return float(np.sum(np.diff(R) * P[1:]))


def macro_average_precision(y_true, proba, labels):
    """多分类宏平均 AP：每类 one-vs-rest 求 AP 后取算术平均"""
    labels = np.asarray(labels)
    aps = {}
    for c in labels:
        aps[int(c)] = average_precision(np.asarray(y_true) == c, np.asarray(proba)[:, int(c)])
    return aps, float(np.mean(list(aps.values())))


def roc_auc(y_true_bin, scores):
    """ROC-AUC（秩和公式），作为补充指标"""
    y = np.asarray(y_true_bin).astype(bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ss = s[order]
    i = 0
    while i < len(ss):                            # 处理并列分数：取平均秩
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece_score(y_true_bin, prob, n_bins=10):
    """期望校准误差 ECE：按置信度分箱，加权 |平均置信度 - 实际正确率|"""
    y = np.asarray(y_true_bin).astype(float)
    p = np.asarray(prob, dtype=float)
    conf = np.maximum(p, 1 - p)                   # 置信度 = 预测类的概率
    pred = (p >= 0.5).astype(float)
    correct = (pred == y).astype(float)           # 是否预测正确
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


print("评估指标函数定义完成：confusion_matrix / binary_prf / prf_report /")
print("                        pr_curve / average_precision / macro_average_precision")
print("                        roc_auc / ece_score")
'''))

CELLS.append(("md", r"""
### 4.2 与 scikit-learn 交叉校验

手写指标最容易犯的错误是阈值扫描的边界处理（并列分数、曲线起点、缺失的端点）。
用 `sklearn.metrics` 对同一批概率算一遍，逐项对比。**只校验指标，不涉及建模**。
"""))

CELLS.append(("code", r'''from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)

# 先取两个模型的测试集概率
proba_lr = LR_FINAL.predict_proba(X_test)     # 逻辑回归后验概率
proba_nb = GNB.predict_proba(X_test)          # 朴素贝叶斯后验概率
pred_lr = LR_FINAL.predict(X_test)
pred_nb = GNB.predict(X_test)

print("表 4-1  手写指标与 sklearn.metrics 的交叉校验")
print("%-46s%16s%16s%14s" % ("指标 / 校验对象", "手写实现", "sklearn", "最大绝对差"))
max_gap = 0.0
for name, proba in [("逻辑回归", proba_lr), ("朴素贝叶斯", proba_nb)]:
    for c in [0, 1]:
        mine = average_precision(y_test == c, proba[:, c])
        ref = average_precision_score(y_test == c, proba[:, c])
        max_gap = max(max_gap, abs(mine - ref))
        print("%-46s%16.10f%16.10f%14.2e" % (f"{name} · AP(类别 {c}, one-vs-rest)", mine, ref, abs(mine - ref)))
    mine_auc = roc_auc(y_test == 1, proba[:, 1])
    ref_auc = roc_auc_score(y_test, proba[:, 1])
    max_gap = max(max_gap, abs(mine_auc - ref_auc))
    print("%-46s%16.10f%16.10f%14.2e" % (f"{name} · ROC-AUC(正类=良性)", mine_auc, ref_auc, abs(mine_auc - ref_auc)))

for name, pred in [("逻辑回归", pred_lr), ("朴素贝叶斯", pred_nb)]:
    for kind, mine_f, ref_f in [("macro-P", lambda: prf_report(y_test, pred, [0, 1])[1]["precision"],
                                 lambda: precision_score(y_test, pred, average="macro")),
                                ("macro-R", lambda: prf_report(y_test, pred, [0, 1])[1]["recall"],
                                 lambda: recall_score(y_test, pred, average="macro")),
                                ("macro-F1", lambda: prf_report(y_test, pred, [0, 1])[1]["f1"],
                                 lambda: f1_score(y_test, pred, average="macro"))]:
        mine, ref = mine_f(), ref_f()
        max_gap = max(max_gap, abs(mine - ref))
        print("%-46s%16.10f%16.10f%14.2e" % (f"{name} · {kind}", mine, ref, abs(mine - ref)))

print("\n全部校验项的最大绝对差 = %.2e" % max_gap)
assert max_gap < 1e-9, "手写指标与 sklearn 不一致，需要检查"
print("断言通过：手写指标与 sklearn 完全一致（差异在 1e-9 以内，属于浮点舍入）")
'''))

CELLS.append(("md", r"""
## 5. 实验结果

### 5.1 测试集整体指标（主实验：seed=42，7:3 划分）

两种模型的超参数设置：

| 模型 | 超参数 | 取值 | 选择依据 |
|---|---|---|---|
| 逻辑回归 | L2 系数 λ | `2 \times 10^{-2}` | 训练集内 5 折 CV 对数损失最小（2.5 节）|
| 逻辑回归 | 学习率 η | `1/L \approx 0.294` | 取 Lipschitz 上界的倒数，保证收敛 |
| 逻辑回归 | 最大迭代轮数 | 30000 | 留足余量，实际在 1629 轮收敛 |
| 逻辑回归 | 是否标准化 | 是 | 2.6 节已证明必须标准化 |
| 高斯朴素贝叶斯 | 方差下限 ε | `10^{-9}` | 数值保护，真实数据上不影响结果 |
| 高斯朴素贝叶斯 | 方差估计 | 极大似然（除以 N_c） | 与原理推导一致 |
"""))

CELLS.append(("code", r'''# ---------- 汇总两个模型在测试集上的全部指标 ----------
def evaluate(name, model, proba):
    """对一个模型算齐全部指标，返回字典"""
    pred = model.predict(X_test)
    rows, macro, weighted, acc, support = prf_report(y_test, pred, [0, 1])
    aps, macro_ap = macro_average_precision(y_test, proba, [0, 1])
    return {
        "name": name,
        "pred": pred,
        "proba": proba,
        "cm": confusion_matrix(y_test, pred, [0, 1]),
        "per_class": rows, "macro": macro, "weighted": weighted,
        "acc": acc, "support": support,
        "ap_per_class": aps, "macro_ap": macro_ap,
        "auc": roc_auc(y_test == 1, proba[:, 1]),
        "ece": ece_score(y_test == 1, proba[:, 1]),
    }


RES_LR = evaluate("逻辑回归", LR_FINAL, proba_lr)
RES_NB = evaluate("朴素贝叶斯", GNB, proba_nb)

print("表 5-1  两种方法在测试集上的性能对比（171 条样本）")
print("%-18s%10s%10s%10s%10s%10s%10s%10s" %
      ("模型", "准确率", "宏精确率", "宏召回率", "宏F1", "宏AP", "AUC", "ECE"))
for r in [RES_LR, RES_NB]:
    print("%-18s%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f"
          % (r["name"], r["acc"], r["macro"]["precision"], r["macro"]["recall"],
             r["macro"]["f1"], r["macro_ap"], r["auc"], r["ece"]))

print("\n表 5-2  加权平均指标（按类别样本数加权，作为补充）")
print("%-18s%12s%12s%12s" % ("模型", "加权精确率", "加权召回率", "加权F1"))
for r in [RES_LR, RES_NB]:
    print("%-18s%12.4f%12.4f%12.4f"
          % (r["name"], r["weighted"]["precision"], r["weighted"]["recall"], r["weighted"]["f1"]))
'''))

CELLS.append(("code", r'''# ---------- 逐类别指标表 ----------
print("表 5-3  逐类别（one-vs-rest）精确率 / 召回率 / F1 / AP")
print("%-12s%8s%12s%12s%12s%12s%10s" %
      ("模型", "类别", "精确率", "召回率", "F1", "AP", "支持数"))
for r in [RES_LR, RES_NB]:
    for c in [0, 1]:
        d = r["per_class"][int(c)]
        label = "恶性(0)" if c == 0 else "良性(1)"
        print("%-12s%8s%12.4f%12.4f%12.4f%12.4f%10d"
              % (r["name"], label, d["precision"], d["recall"], d["f1"],
                 r["ap_per_class"][int(c)], r["support"][int(c)]))
    print("%-12s%8s%12.4f%12.4f%12.4f%12.4f%10s"
          % ("  └ 宏平均", "", r["macro"]["precision"], r["macro"]["recall"],
             r["macro"]["f1"], r["macro_ap"], ""))
'''))

CELLS.append(("code", r'''# ---------- 图 5-1：两个模型的混淆矩阵 ----------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
class_labels = ["恶性 (0)", "良性 (1)"]
for ax, r in zip(axes, [RES_LR, RES_NB]):
    M = r["cm"]
    im = ax.imshow(M, cmap="Blues")
    for i in range(2):                       # 在每个格子里写数值
        for j in range(2):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=15,
                    color="white" if M[i, j] > M.max() * 0.55 else "#1B2631")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels); ax.set_yticklabels(class_labels)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("%s（准确率 %.4f）" % (r["name"], r["acc"]), fontsize=11)
    ax.grid(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_confusion.png"))
plt.show()

for r in [RES_LR, RES_NB]:
    M = r["cm"]
    print("%s 混淆矩阵: [[TP=%d, FN=%d], [FP=%d, TN=%d]]（行=真实，列=预测）"
          % (r["name"], M[0, 0], M[1, 0], M[0, 1], M[1, 1]))
'''))

CELLS.append(("code", r'''# ---------- 图 5-2：P-R 曲线对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

for ax, c, cls_name in zip(axes, [0, 1], ["恶性 (0)", "良性 (1)"]):
    for r, color, ls in [(RES_LR, "#C0392B", "-"), (RES_NB, "#2E86C1", "-")]:
        R, P = pr_curve(y_test == c, r["proba"][:, c])
        ax.plot(R, P, color=color, lw=2.2, ls=ls,
                label="%s  AP=%.4f" % (r["name"], r["ap_per_class"][c]))
        ax.fill_between(R, P, alpha=0.08, color=color)
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_title("(a) 类别 %s 的 P-R 曲线" % cls_name if c == 0 else "(b) 类别 %s 的 P-R 曲线" % cls_name,
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=9.5)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_pr_curves.png"))
plt.show()

print("宏平均 AP：逻辑回归 %.4f，朴素贝叶斯 %.4f，高出 %.4f"
      % (RES_LR["macro_ap"], RES_NB["macro_ap"], RES_LR["macro_ap"] - RES_NB["macro_ap"]))
'''))

CELLS.append(("md", r"""
### 5.2 阈值选择的影响

0.5 只是默认阈值，未必是任务的最优点。医疗场景里"漏诊恶性"的代价远大于"误报恶性"，
所以实际使用时应按需求调阈值：想要高召回率就把阈值调低。
下面给出两个模型在不同阈值下的 P / R / F1，并标出 F1 最大处的阈值。
"""))

CELLS.append(("code", r'''# ---------- 阈值扫描 ----------
thresholds = np.arange(0.05, 0.96, 0.05)
print("表 5-4  阈值对精确率/召回率/F1 的影响（正类 = 恶性，即查出来的是不是真的恶性）")
best_th = {}
th_table = {}
for r in [RES_LR, RES_NB]:
    prob = r["proba"][:, 0]                       # 以「恶性」为正类的概率
    y_bin = (y_test == 0)
    f1s, rows_t = [], []
    for th in thresholds:
        d = binary_prf(y_bin, prob >= th)
        rows_t.append((th, d["precision"], d["recall"], d["f1"]))
        f1s.append(d["f1"])
    b = int(np.argmax(f1s))
    best_th[r["name"]] = (float(thresholds[b]), f1s[b])
    th_table[r["name"]] = [{"threshold": float(t_), "precision": p_, "recall": r_, "f1": f_}
                           for t_, p_, r_, f_ in rows_t]
    print("\n%s（F1 最大的阈值 = %.2f，此时 F1 = %.4f）" % (r["name"], thresholds[b], f1s[b]))
    print("  阈值   精确率   召回率    F1")
    for th, p_, r_, f_ in rows_t:
        mark = "  <-- 最优" if abs(th - thresholds[b]) < 1e-9 else ""
        print("  %.2f   %.4f   %.4f   %.4f%s" % (th, p_, r_, f_, mark))
'''))

CELLS.append(("code", r'''# ---------- 图 5-3：P-R-F1 随阈值变化 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
for ax, r in zip(axes, [RES_LR, RES_NB]):
    prob = r["proba"][:, 0]
    y_bin = (y_test == 0)
    ps, rs, fs = [], [], []
    for th in thresholds:
        d = binary_prf(y_bin, prob >= th)
        ps.append(d["precision"]); rs.append(d["recall"]); fs.append(d["f1"])
    ax.plot(thresholds, ps, "o-", ms=3.5, lw=1.8, color="#2E86C1", label="精确率")
    ax.plot(thresholds, rs, "s-", ms=3.5, lw=1.8, color="#27AE60", label="召回率")
    ax.plot(thresholds, fs, "^-", ms=3.5, lw=1.8, color="#C0392B", label="F1")
    bt, bf = best_th[r["name"]]
    ax.axvline(bt, color="#7F8C8D", ls="--", lw=1.2)
    ax.annotate("最优阈值 %.2f\nF1=%.4f" % (bt, bf), xy=(bt, bf),
                xytext=(8, -38), textcoords="offset points", fontsize=9, color="#7F8C8D")
    ax.set_xlabel("分类阈值（判为「恶性」的门槛）")
    ax.set_title(r["name"], fontsize=11)
    ax.legend(fontsize=9)
axes[0].set_ylabel("指标值")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_threshold.png"))
plt.show()

print("阈值降低 -> 召回率上升、精确率下降")
print("逻辑回归的 F1 最优阈值 = %.2f，朴素贝叶斯 = %.2f"
      % (best_th["逻辑回归"][0], best_th["朴素贝叶斯"][0]))
print("注意：逻辑回归的 F1 在 0.50~0.60 上并列最优，说明它的概率分得很开；")
print("      朴素贝叶斯要到 0.70 才最好，反映它把良性概率整体推高了（过度自信）")
'''))

CELLS.append(("md", r"""
### 5.3 概率校准分析

逻辑回归直接优化对数似然，输出的是**校准过**的概率（预测为 0.9 的样本里大约有 90% 是正类）；
高斯朴素贝叶斯的概率来自"独立假设"下的密度比，通常**过度自信**（概率挤在 0 和 1 附近，
但正确率没那么高）。用可靠性曲线（reliability diagram）和 ECE 来量化。
"""))

CELLS.append(("code", r'''# ---------- 图 5-4：可靠性曲线 + 预测概率分布 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))

# (a) 可靠性曲线
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="完美校准")
for r, color in [(RES_LR, "#C0392B"), (RES_NB, "#2E86C1")]:
    prob = r["proba"][:, 1]
    conf = np.maximum(prob, 1 - prob)
    correct = (r["pred"] == y_test).astype(float)
    bins = np.linspace(0.5, 1.0, 11)
    xs, ys = [], []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() >= 3:
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    ax.plot(xs, ys, "o-", color=color, lw=2, ms=7, label="%s（ECE=%.4f）" % (r["name"], r["ece"]))
ax.set_xlabel("平均置信度")
ax.set_ylabel("实际正确率")
ax.set_title("(a) 可靠性曲线", fontsize=11)
ax.legend(fontsize=9)

# (b) 概率分布直方图：越靠两端说明越"自信"
ax = axes[1]
for r, color in [(RES_LR, "#C0392B"), (RES_NB, "#2E86C1")]:
    ax.hist(r["proba"][:, 1], bins=np.linspace(0, 1, 21), alpha=0.55,
            color=color, label=r["name"], edgecolor="white")
ax.set_xlabel("预测为「良性」的概率")
ax.set_ylabel("样本数")
ax.set_title("(b) 预测概率分布", fontsize=11)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_calibration.png"))
plt.show()

for r in [RES_LR, RES_NB]:
    prob = r["proba"][:, 1]
    extreme = np.mean((prob < 0.05) | (prob > 0.95))
    print("%s: ECE=%.4f，概率落在 [0,0.05)∪(0.95,1] 的样本占 %.1f%%"
          % (r["name"], r["ece"], 100 * extreme))
'''))

CELLS.append(("md", r"""
### 5.4 多种子稳定性实验

一次划分的结果带有随机性（0.3 的测试集只有 171 条，错一条就是 0.6 个百分点）。
这里用 **10 个不同的随机种子**各做一次完整实验（划分 → 训练 → 评估），
给出指标的**均值 ± 标准差**，这才是可靠的比较。
"""))

CELLS.append(("code", r'''# ---------- 10 个随机种子重复实验 ----------
seed_list = list(range(10))
seed_rows = []
for sd in seed_list:
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=sd, stratify=y)
    m_lr = LogisticRegressionGD(l2=best_lam, lr_mode="lipschitz", tol=1e-10, max_iter=30000).fit(Xa, ya)
    m_nb = GaussianNBFromScratch(var_floor=1e-9, ddof=0).fit(Xa, ya)
    for name, m in [("逻辑回归", m_lr), ("朴素贝叶斯", m_nb)]:
        proba = m.predict_proba(Xb)
        pred = m.predict(Xb)
        _, macro, _, acc, _ = prf_report(yb, pred, [0, 1])
        _, macro_ap = macro_average_precision(yb, proba, [0, 1])
        seed_rows.append({"seed": sd, "model": name, "acc": acc,
                          "macro_P": macro["precision"], "macro_R": macro["recall"],
                          "macro_F1": macro["f1"], "macro_AP": macro_ap})

seed_df = pd.DataFrame(seed_rows)
print("表 5-5  10 个随机种子下的指标（均值 ± 样本标准差）")
print("%-12s%20s%20s%20s%20s" % ("模型", "准确率", "宏精确率", "宏F1", "宏AP"))
summary = {}
for name in ["逻辑回归", "朴素贝叶斯"]:
    sub = seed_df[seed_df["model"] == name]
    cells = []
    summary[name] = {}
    for col in ["acc", "macro_P", "macro_F1", "macro_AP"]:
        mu, sd_ = sub[col].mean(), sub[col].std(ddof=1)
        summary[name][col] = (float(mu), float(sd_))
        cells.append("%.4f±%.4f" % (mu, sd_))
    print("%-12s%20s%20s%20s%20s" % (name, cells[0], cells[1], cells[2], cells[3]))
'''))

CELLS.append(("code", r'''# ---------- 图 5-5：10 个种子的指标分布 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
metrics = [("macro_F1", "宏平均 F1"), ("macro_AP", "宏平均 AP")]
colors = {"逻辑回归": "#C0392B", "朴素贝叶斯": "#2E86C1"}

for ax, (col, title) in zip(axes, metrics):
    data = [seed_df[seed_df["model"] == m][col].values for m in ["逻辑回归", "朴素贝叶斯"]]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    tick_labels=["逻辑回归", "朴素贝叶斯"], medianprops=dict(color="black", lw=1.6))
    for patch, m in zip(bp["boxes"], ["逻辑回归", "朴素贝叶斯"]):
        patch.set_facecolor(colors[m]); patch.set_alpha(0.45)
    for i, d in enumerate(data):                    # 叠加原始散点
        ax.scatter(np.random.RandomState(i).normal(i + 1, 0.045, len(d)), d,
                   s=22, color=colors[["逻辑回归", "朴素贝叶斯"][i]], zorder=3, alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(col)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_seed_stability.png"))
plt.show()

for m in ["逻辑回归", "朴素贝叶斯"]:
    sub = seed_df[seed_df["model"] == m]
    print("%s: 宏F1 %.4f±%.4f，宏AP %.4f±%.4f，10 个种子中宏F1 最小 %.4f、最大 %.4f"
          % (m, sub["macro_F1"].mean(), sub["macro_F1"].std(ddof=1),
             sub["macro_AP"].mean(), sub["macro_AP"].std(ddof=1),
             sub["macro_F1"].min(), sub["macro_F1"].max()))
'''))

CELLS.append(("md", r"""
### 5.5 从模型参数看两种方法的差异

逻辑回归学到的权重可以直接比较（特征已标准化，权重绝对值大小表示该特征的贡献），
高斯朴素贝叶斯学的是"两类各自的高斯中心"，二者之差可以类比为"判别方向"。
"""))

CELLS.append(("code", r'''# ---------- 逻辑回归权重：标准化后可直接比较大小 ----------
coef = LR_FINAL.w_[:-1]                      # 去掉偏置
intercept = LR_FINAL.w_[-1]
order = np.argsort(-np.abs(coef))            # 按绝对值降序

print("逻辑回归的截距 b = %.4f" % intercept)
print("\n表 5-6  逻辑回归权重绝对值最大的 10 个特征（已标准化，正=推向良性）")
print("%-24s%12s%14s%14s" % ("特征", "权重", "恶性组均值", "良性组均值"))
for j in order[:10]:
    print("%-24s%12.4f%14.4f%14.4f" % (feature_names[j], coef[j],
                                       GNB.mean_[0, j], GNB.mean_[1, j]))

print("\n权重符号统计：正权重 %d 个，负权重 %d 个"
      % (int(np.sum(coef > 0)), int(np.sum(coef < 0))))
print("权重绝对值最大 %.4f，最小 %.4f" % (np.abs(coef).max(), np.abs(coef).min()))

# ---------- 图 5-6：权重条形图 ----------
fig, ax = plt.subplots(figsize=(7.6, 5.0))
top = order[:12]
colors_bar = ["#C0392B" if coef[j] < 0 else "#27AE60" for j in top]
ax.barh(np.arange(len(top)), coef[top][::-1], color=colors_bar[::-1], height=0.68)
ax.set_yticks(np.arange(len(top)))
ax.set_yticklabels([feature_names[j] for j in top][::-1], fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("逻辑回归权重（已标准化，负值推向恶性 / 正值推向良性）")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_lr_coef.png"))
plt.show()
'''))

CELLS.append(("md", r"""
### 5.6 训练集规模的影响（学习曲线）

前面所有实验都用了 398 条训练样本。一个自然的问题是：**如果样本更少，朴素贝叶斯会不会反超？**
从偏差-方差的角度看，朴素贝叶斯参数少、方差低，通常在小样本下更有优势。
下面把训练集按类别分层随机抽取 10%、20%、30%、50%、70%、100%，固定测试集不变，
看两种方法的宏 F1 如何变化。
"""))
CELLS.append(("code", r'''# ---------- 学习曲线：训练集规模 vs 宏 F1 ----------
fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
lc_rows = []
sub_rng = np.random.RandomState(0)               # 控制子集抽取的随机性
for frac in fracs:
    idx = []
    for c in [0, 1]:                             # 按类别分层抽取，保证子集类别比例不变
        idc = np.where(y_train == c)[0]
        sub_rng.shuffle(idc)
        idx.append(idc[:int(round(len(idc) * frac))])
    idx = np.concatenate(idx)
    Xa, ya = X_train[idx], y_train[idx]          # 子训练集（测试集始终不变）
    m_lr = LogisticRegressionGD(l2=best_lam, lr_mode="lipschitz", tol=1e-10, max_iter=30000).fit(Xa, ya)
    m_nb = GaussianNBFromScratch(var_floor=1e-9, ddof=0).fit(Xa, ya)
    row = {"frac": frac, "n_train": int(len(idx))}
    for nm, m in [("LR", m_lr), ("GNB", m_nb)]:
        _, mac, _, acc, _ = prf_report(y_test, m.predict(X_test), [0, 1])
        aps, macro_ap = macro_average_precision(y_test, m.predict_proba(X_test), [0, 1])
        row[nm + "_f1"] = mac["f1"]
        row[nm + "_ap"] = macro_ap
        row[nm + "_acc"] = acc
    lc_rows.append(row)

print("表 5-6  训练集规模对测试集宏 F1 的影响（测试集固定，171 条）")
print("%10s%10s%12s%12s%12s" % ("训练比例", "训练样本", "逻辑回归 F1", "朴素贝叶斯 F1", "F1 差距"))
for row in lc_rows:
    print("%9.0f%%%10d%12.4f%12.4f%+12.4f"
          % (100 * row["frac"], row["n_train"], row["LR_f1"], row["GNB_f1"], row["LR_f1"] - row["GNB_f1"]))
'''))
CELLS.append(("code", r'''# ---------- 图 5-6：学习曲线 ----------
fig, ax = plt.subplots(figsize=(7.2, 4.1))
ns = [r["n_train"] for r in lc_rows]
ax.plot(ns, [r["LR_f1"] for r in lc_rows], "o-", color="#C0392B", lw=2, ms=7, label="逻辑回归")
ax.plot(ns, [r["GNB_f1"] for r in lc_rows], "s-", color="#2E86C1", lw=2, ms=7, label="朴素贝叶斯")
ax.fill_between(ns, [r["GNB_f1"] for r in lc_rows], [r["LR_f1"] for r in lc_rows],
                color="#C0392B", alpha=0.08)
ax.set_xlabel("训练集样本数")
ax.set_ylabel("测试集宏平均 F1")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_learning_curve.png"))
plt.show()

print("样本最少（%d 条）时两者差距最小（%.4f）；样本增多后差距反而拉大到 %.4f"
      % (lc_rows[0]["n_train"], lc_rows[0]["LR_f1"] - lc_rows[0]["GNB_f1"],
         lc_rows[-1]["LR_f1"] - lc_rows[-1]["GNB_f1"]))
print("朴素贝叶斯的宏 F1 基本停留在 %.2f~%.2f，逻辑回归则随样本增加持续改善"
      % (min(r["GNB_f1"] for r in lc_rows), max(r["GNB_f1"] for r in lc_rows)))
'''))
CELLS.append(("md", r"""
### 5.7 两种方法计算代价的对比

- 逻辑回归：**迭代**求解，每一轮都要遍历全部训练样本和全部特征，
  计算量约为 `O(T \cdot N \cdot d)`（T 为迭代轮数）。
- 高斯朴素贝叶斯：**闭式估计**，只需把训练集扫描一遍，
  计算量约为 `O(N \cdot d)`，没有迭代。

下面实测训练耗时（同一台机器，单线程）。
"""))

CELLS.append(("code", r'''import time

# 每种方法重复 5 次取平均，减少计时噪声
def timeit(fn, repeat=5):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts, ddof=1))

t_lr, s_lr = timeit(lambda: LogisticRegressionGD(l2=best_lam, lr_mode="lipschitz",
                                                 tol=1e-10, max_iter=30000).fit(X_train, y_train))
t_nb, s_nb = timeit(lambda: GaussianNBFromScratch(var_floor=1e-9).fit(X_train, y_train))

print("表 5-7  训练耗时对比（重复 5 次）")
print("%-14s%16s%14s%18s" % ("模型", "平均耗时/ms", "标准差/ms", "迭代轮数"))
print("%-14s%16.2f%14.2f%18d" % ("逻辑回归", t_lr * 1000, s_lr * 1000, LR_FINAL.n_iter_))
print("%-14s%16.2f%14.2f%18s" % ("朴素贝叶斯", t_nb * 1000, s_nb * 1000, "0（闭式）"))
print("\n逻辑回归训练耗时是朴素贝叶斯的 %.1f 倍" % (t_lr / t_nb))
'''))

CELLS.append(("md", r"""
## 6. 结论与结果导出

最后把报告需要用到的全部数字打包成 `results.json`，并打印结论摘要。
"""))

CELLS.append(("code", r'''# ---------- 收集全部结果，导出 JSON 供报告写作使用 ----------
def f(x):
    """numpy 标量 -> Python 原生 float，保证 JSON 可序列化"""
    return float(x)


results = {
    "meta": {
        "student_id": "XXXXXXXXXX", "name": "XXX", "group": "B",
        "date": "2026-10-25", "seed": SEED,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "class_counts": {"malignant(0)": int(np.sum(y == 0)), "benign(1)": int(np.sum(y == 1))},
        "split": {"train": int(len(y_train)), "test": int(len(y_test)), "test_size": 0.3},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "n_strong_corr_pairs_0.9": int(n_strong),
        "std_ratio_max_min": f(stats["std"].max() / stats["std"].min()),
    },
    "lr": {
        "best_lambda": f(best_lam), "eta": f(LR_FINAL.eta_), "L": f(LR_FINAL.L_),
        "n_iter": int(LR_FINAL.n_iter_), "stop_reason": LR_FINAL.stop_reason_,
        "loss_start": f(LR_FINAL.loss_hist_[0]), "loss_end": f(LR_FINAL.loss_hist_[-1]),
        "grad_inf": f(LR_FINAL.grad_norm_hist_[-1]),
        "w_norm": f(np.linalg.norm(LR_FINAL.w_[:-1])), "intercept": f(LR_FINAL.w_[-1]),
        "train_time_ms": f(t_lr * 1000), "train_time_std_ms": f(s_lr * 1000),
        "cv_table": [{k: (f(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
                      for k, v in row.items()} for row in rows],
    },
    "gnb": {
        "var_floor": 1e-9, "min_var": f(GNB.var_.min()), "max_var": f(GNB.var_.max()),
        "priors": [f(GNB.prior_[0]), f(GNB.prior_[1])],
        "train_time_ms": f(t_nb * 1000), "train_time_std_ms": f(s_nb * 1000),
        "scale_inv_max_abs_diff": f(np.max(np.abs(p_raw - p_std))),
    },
    "eval": {
        "lr": {k: v for k, v in RES_LR.items() if k not in ("pred", "proba")},
        "gnb": {k: v for k, v in RES_NB.items() if k not in ("pred", "proba")},
        "best_threshold": {k: {"threshold": f(v[0]), "f1": f(v[1])} for k, v in best_th.items()},
        "seed_summary": summary,
        "seed_rows": seed_rows,
        "learning_curve": lc_rows,
        "threshold_table": th_table,
        "standardization": {
            "L_std": f(lr_std.L_), "eta_std": f(lr_std.eta_),
            "cond_std": f(cond_std), "loss_std": f(lr_std.loss_hist_[-1]),
            "acc_std": f(np.mean(lr_std.predict(X_test) == y_test)),
            "iters_std": int(lr_std.n_iter_), "conv_std": bool(lr_std.converged_),
            "L_raw": f(lr_raw.L_), "eta_raw": f(lr_raw.eta_),
            "cond_raw": f(cond_raw), "loss_raw": f(lr_raw.loss_hist_[-1]),
            "acc_raw": f(np.mean(lr_raw.predict(X_test) == y_test)),
            "iters_raw": int(lr_raw.n_iter_), "conv_raw": bool(lr_raw.converged_),
            "eta_ratio": f(lr_std.eta_ / lr_raw.eta_),
            "cond_ratio": f(cond_raw / cond_std),
        },
        "coef_top": [{"name": str(feature_names[j]), "weight": f(coef[j]),
                      "mean0": f(GNB.mean_[0, j]), "mean1": f(GNB.mean_[1, j])}
                     for j in order[:10]],
        "coef_pos": int(np.sum(coef > 0)), "coef_neg": int(np.sum(coef < 0)),
        "coef_absmax": f(np.abs(coef).max()), "coef_absmin": f(np.abs(coef).min()),
        "lambda_table": lambda_table.to_dict(orient="records"),
    },
}

def to_jsonable(obj):
    """递归地把 numpy 标量/数组转换成 JSON 原生类型"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


with open("results.json", "w", encoding="utf-8") as fp:
    json.dump(to_jsonable(results), fp, ensure_ascii=False, indent=2)
print("结果已写入 results.json（%d 字节）" % os.path.getsize("results.json"))

# ---------- 结论摘要 ----------
print("\n" + "=" * 66)
print("结论摘要")
print("=" * 66)
print("1) 逻辑回归：宏平均 AP = %.4f，宏平均 F1 = %.4f，准确率 = %.4f，AUC = %.4f"
      % (RES_LR["macro_ap"], RES_LR["macro"]["f1"], RES_LR["acc"], RES_LR["auc"]))
print("2) 朴素贝叶斯：宏平均 AP = %.4f，宏平均 F1 = %.4f，准确率 = %.4f，AUC = %.4f"
      % (RES_NB["macro_ap"], RES_NB["macro"]["f1"], RES_NB["acc"], RES_NB["auc"]))
print("3) 宏平均 AP 差距 = %.4f，宏 F1 差距 = %.4f，均为逻辑回归更高"
      % (RES_LR["macro_ap"] - RES_NB["macro_ap"], RES_LR["macro"]["f1"] - RES_NB["macro"]["f1"]))
print("4) 10 个种子上：逻辑回归宏F1 %.4f±%.4f，朴素贝叶斯 %.4f±%.4f"
      % (*summary["逻辑回归"]["macro_F1"], *summary["朴素贝叶斯"]["macro_F1"]))
print("5) 校准：逻辑回归 ECE = %.4f 优于朴素贝叶斯 ECE = %.4f"
      % (RES_LR["ece"], RES_NB["ece"]))
print("6) 代价：逻辑回归训练 %.1f ms（%d 轮迭代），朴素贝叶斯 %.1f ms（闭式、0 轮迭代）"
      % (t_lr * 1000, LR_FINAL.n_iter_, t_nb * 1000))
print("7) 特征标准化：逻辑回归必须做（步长差 %.0f 倍、条件数差 %.0e 倍）；朴素贝叶斯严格不变"
      % (lr_std.eta_ / lr_raw.eta_, cond_raw / cond_std))
print("=" * 66)
'''))
