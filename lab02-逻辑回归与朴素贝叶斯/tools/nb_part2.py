# -*- coding: utf-8 -*-
"""Notebook 第 2 部分：手写逻辑回归"""

CELLS = []

CELLS.append(("md", r"""
## 2. 逻辑回归的手写实现

### 2.1 算法原理

#### （1）模型形式

二分类逻辑回归把特征的线性组合 `z = w^T x + b` 通过 **sigmoid 函数**压缩到 (0,1)，
作为"样本属于正类"的概率：

$$ \hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \mathbf{w}^{\top}\mathbf{x} + b $$

写成向量形式时，把偏置 b 并入权重、在特征矩阵右侧补一列 1，
即 `\tilde{\mathbf{x}} = [\mathbf{x}, 1]`、`\mathbf{w} = [\mathbf{w}, b]`，
模型统一为 `\hat{p} = \sigma(\tilde{\mathbf{x}}^{\top}\mathbf{w})`。

#### （2）目标函数：极大似然 → 负对数似然

单个样本的似然写成 Bernoulli 分布：

$$ P(y \mid \mathbf{x}) = \hat{p}^{\,y} (1 - \hat{p})^{\,1-y} $$

N 个样本独立时总似然为 `\prod_i P(y_i \mid \mathbf{x}_i)`。两边取对数（不改变极值点，
且把连乘变成求和，避免下溢）：

$$ \ell(\mathbf{w}) = \sum_{i=1}^{N} \left[ y_i \ln \hat{p}_i + (1 - y_i)\ln(1 - \hat{p}_i) \right] $$

极大似然估计就是**最大化** `\ell`，等价于**最小化负对数似然**（即交叉熵）：

$$ J(\mathbf{w}) = -\frac{1}{N}\ell(\mathbf{w}) + \frac{\lambda}{2}\lVert \mathbf{w}_{1:d} \rVert^2
= -\frac{1}{N}\sum_{i=1}^{N}\left[ y_i \ln \hat{p}_i + (1-y_i)\ln(1-\hat{p}_i) \right] + \frac{\lambda}{2}\sum_{j=1}^{d} w_j^2 $$

后面一项是 **L2 正则**，系数 `\lambda` 越大，权重被压得越接近 0。
在 2.5 节会说明：本数据集**近似线性可分**，不加正则时极大似然解根本不存在，
必须靠 L2 才能得到唯一、有限的解。

#### （3）梯度

对 `z_i` 求导：`\frac{\partial}{\partial z} \left[y\ln\sigma(z) + (1-y)\ln(1-\sigma(z))\right] = y - \sigma(z)`，
再用链式法则对 `\mathbf{w}` 求导（`\partial z_i / \partial \mathbf{w} = \tilde{\mathbf{x}}_i`），得到

$$ \nabla J(\mathbf{w}) = \frac{1}{N}\tilde{\mathbf{X}}^{\top}(\hat{\mathbf{p}} - \mathbf{y}) + \lambda \mathbf{w}_{1:d}, \qquad
\frac{\partial J}{\partial b} = \frac{1}{N}\sum_{i=1}^{N}(\hat{p}_i - y_i) $$

#### （4）求解：全批量梯度下降

$$ \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla J(\mathbf{w}^{(t)}) $$

其中 `\eta` 为学习率。由于 `J` 是凸函数（负对数似然为凸，L2 为严格凸），
梯度下降收敛到的驻点就是全局最优。

**收敛判据**：梯度无穷范数 `\lVert \nabla J \rVert_\infty < \mathrm{tol}`，
或相邻两轮损失相对变化小于 tol，或达到最大迭代轮数。

#### （5）步长的理论上界

`J` 的梯度是 Lipschitz 连续的，常数

$$ L = \frac{\lambda_{\max}(\tilde{\mathbf{X}}^{\top}\tilde{\mathbf{X}}/N)}{4} + \lambda $$

只要取 `\eta = 1/L`，梯度下降就能保证单调下降且收敛。
这个式子会在 2.6 节用来解释"为什么标准化前后需要的步长差几个数量级"。
"""))

CELLS.append(("md", r"""
### 2.2 数值稳定实现

朴素实现里有三处会数值溢出：

1. `\exp(-z)` 在 z 为大负数时上溢 → 在 sigmoid 中按符号分段计算；
2. `\ln(1 + e^{-z})` 在 z 为大负数时会退化成 `\ln(1)=0` 而损失信息 → 用 `np.logaddexp`；
3. 概率归一化时的 `\exp` 求和会上下溢 → 用 **log-sum-exp** 技巧（先减最大值）。

这三段是后面所有代码的基础。
"""))

CELLS.append(("code", r'''# ============================================================ 数值稳定工具
def sigmoid(z):
    """数值稳定的 sigmoid：按符号分段，避免 exp(-z) 在 z 很小时上溢"""
    z = np.asarray(z, dtype=float)          # 统一转成浮点数组
    out = np.empty_like(z)                  # 预分配输出数组
    pos = z >= 0                            # 正半轴：直接算 1/(1+exp(-z))，exp 的参数不会为正
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])                    # 负半轴：改用 exp(z)/(1+exp(z))，参数同样不会为正
    out[~pos] = ez / (1.0 + ez)
    return out


def log_sigmoid(z):
    """ln sigma(z) = -ln(1+exp(-z))；用 logaddexp 保证 z 为极大负数时不丢精度"""
    return -np.logaddexp(0.0, -np.asarray(z, dtype=float))


def logsumexp(a, axis=None, keepdims=False):
    """log-sum-exp 技巧：先减每行最大值再取对数，避免 exp 上溢/下溢"""
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)           # 每行最大值
    m_safe = np.where(np.isfinite(m), m, 0.0)         # 最大值是 -inf 时退回 0，避免 nan
    s = m_safe + np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True))
    if not keepdims:
        s = np.squeeze(s, axis=axis)
    return s


# ---------- 自检：与朴素公式在安全区间内应完全一致 ----------
z_probe = np.array([-800.0, -30.0, -1.0, 0.0, 1.0, 30.0, 800.0])
print("z             :", z_probe)
print("sigmoid(z)    :", sigmoid(z_probe))
print("log_sigmoid(-800) =", log_sigmoid(-800.0), "（有限值，未溢出）")
print("logsumexp([1000, 1001]) =", logsumexp(np.array([1000.0, 1001.0])))
print("分段函数无 nan/inf :", bool(np.all(np.isfinite(sigmoid(z_probe)))))
'''))

CELLS.append(("md", r"""
### 2.3 手写标准化器

三行公式：`x' = (x - \mu) / \sigma`，其中 `\mu`、`\sigma` **只在训练集上估计**。
测试集只调用 `transform`，不重新 `fit`，否则测试集的分布信息会漏进模型。
"""))

CELLS.append(("code", r'''class StandardScalerFromScratch:
    """z-score 标准化：均值/标准差只在训练集上估计，防止信息泄露"""

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)                      # 每列均值 mu（按列）
        sd = X.std(axis=0, ddof=0)                       # 每列总体标准差 sigma
        self.scale_ = np.where(sd > 0, sd, 1.0)          # 常数列单独处理，避免除零
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_   # 用训练集参数变换

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# 自检：训练集变换后每列均值≈0、标准差≈1
sc_demo = StandardScalerFromScratch()
Xtr_demo = sc_demo.fit_transform(X_train)
print("标准化后训练集每列均值的绝对值最大值: %.2e" % np.abs(Xtr_demo.mean(axis=0)).max())
print("标准化后训练集每列标准差的最大偏差  : %.2e" % np.abs(Xtr_demo.std(axis=0) - 1).max())
print("测试集只做 transform，其均值不再为 0（正常现象），前 3 列 =",
      np.round(sc_demo.transform(X_test).mean(axis=0)[:3], 4))
'''))

CELLS.append(("md", r"""
### 2.4 手写逻辑回归类

代码结构（对应报告第三节）：

| 方法 | 作用 |
|---|---|
| `__init__` | 保存超参数：学习率 lr、最大迭代轮数 max_iter、容忍度 tol、L2 系数 l2、是否标准化 |
| `_add_bias` | 在特征矩阵右侧补一列 1，把偏置 b 并入权重向量 |
| `lipschitz` | 计算梯度 Lipschitz 常数，用于确定理论上界步长 |
| `_loss_grad` | 一次算出负对数似然 + L2 项、梯度、概率 p（三者在同一轮里复用） |
| `fit` | 标准化 → 初始化 → 迭代更新 → 记录损失/梯度范数曲线 |
| `fit_irls` | 用牛顿法（IRLS）解**同一个** MLE，仅用于验证梯度下降收敛到了最优 |
| `predict_proba` / `predict` | 输出概率与类别 |
"""))

CELLS.append(("code", r'''class LogisticRegressionGD:
    """二分类逻辑回归：极大似然估计 + 全批量梯度下降求解（手写，不依赖 sklearn）"""

    def __init__(self, lr=0.5, max_iter=20000, tol=1e-10, l2=1e-2,
                 standardize=True, lr_mode="fixed"):
        self.lr = lr                    # 学习率 eta（lr_mode="fixed" 时生效）
        self.max_iter = max_iter        # 最大迭代轮数
        self.tol = tol                  # 收敛容忍度
        self.l2 = l2                    # L2 正则系数 lambda
        self.standardize = standardize  # 是否先做 z-score 标准化
        self.lr_mode = lr_mode          # "fixed"=固定步长；"lipschitz"=按 1/L 取步长

    # ------------------------------------------------ 数据准备
    def _prep(self, X):
        """按训练集参数做标准化（预测阶段绝不能重新 fit）"""
        X = np.asarray(X, float)
        if self.standardize:
            return self.scaler_.transform(X)
        return X

    @staticmethod
    def _add_bias(X):
        """在最后一列补 1，把偏置 b 并入权重向量 w"""
        return np.hstack([X, np.ones((X.shape[0], 1))])

    @staticmethod
    def lipschitz(Xa):
        """梯度 Lipschitz 常数 L = lambda_max(X^T X / N) / 4"""
        M = (Xa.T @ Xa) / Xa.shape[0]                  # 特征二阶矩矩阵（含偏置列）
        return float(np.linalg.eigvalsh(M)[-1] / 4.0)  # 最大特征值 / 4

    # ------------------------------------------------ 损失与梯度
    def _loss_grad(self, Xa, y):
        """返回 (平均负对数似然 + L2 项, 梯度, 正类概率 p)"""
        z = Xa @ self.w_                                        # 线性组合 z = Xw
        # ln sigma(z) 与 ln sigma(-z)，等价于 ln p 与 ln(1-p)，避免 log(0)
        ll = y * log_sigmoid(z) + (1.0 - y) * log_sigmoid(-z)   # 每个样本的对数似然
        nll = -np.mean(ll)                                      # 平均负对数似然
        reg = 0.5 * self.l2 * np.sum(self.w_[:-1] ** 2)          # L2 项（偏置不参与）
        p = sigmoid(z)                                          # 正类概率
        grad = (Xa.T @ (p - y)) / Xa.shape[0]                   # 梯度主项 (1/N)X^T(p-y)
        grad[:-1] += self.l2 * self.w_[:-1]                     # L2 的梯度 lambda*w
        return nll + reg, grad, p

    # ------------------------------------------------ 训练
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if self.standardize:                                     # 只在训练集上拟合标准化参数
            self.scaler_ = StandardScalerFromScratch().fit(X)
            Xs = self.scaler_.transform(X)
        else:
            self.scaler_ = None
            Xs = X
        Xa = self._add_bias(Xs)                                  # 增广矩阵
        self.w_ = np.zeros(Xa.shape[1])                          # 权重初始化为 0
        L = self.lipschitz(Xa) + self.l2                         # 梯度 Lipschitz 常数
        self.L_ = L
        eta = 1.0 / L if self.lr_mode == "lipschitz" else self.lr  # 步长
        self.eta_ = eta
        self.loss_hist_, self.grad_norm_hist_ = [], []           # 记录收敛过程
        prev = np.inf
        for it in range(self.max_iter):
            J, grad, _ = self._loss_grad(Xa, y)                  # 前向 + 反向
            self.loss_hist_.append(J)                            # 记录当前损失
            self.grad_norm_hist_.append(float(np.max(np.abs(grad))))  # 记录梯度无穷范数
            if np.max(np.abs(grad)) < self.tol:                  # 判据一：梯度足够小
                self.n_iter_, self.stop_reason_ = it, "grad"
                break
            self.w_ = self.w_ - eta * grad                       # 参数更新
            if abs(prev - J) < self.tol * max(1.0, abs(J)):      # 判据二：损失几乎不变
                self.n_iter_, self.stop_reason_ = it + 1, "loss"
                break
            prev = J
        else:
            self.n_iter_, self.stop_reason_ = self.max_iter, "max_iter"  # 判据三：用满轮数
        self.converged_ = self.stop_reason_ in ("grad", "loss")
        return self

    # ------------------------------------------------ 预测
    def decision_function(self, X):
        """返回线性组合 z（>=0 判为正类）"""
        return self._add_bias(self._prep(X)) @ self.w_

    def predict_proba(self, X):
        """返回 (N,2) 概率矩阵 [P(y=0|x), P(y=1|x)]"""
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """取 0.5 为阈值，等价于判断 z >= 0"""
        return (self.decision_function(X) >= 0).astype(int)

    # ------------------------------------------------ 牛顿法/IRLS（仅作收敛性校验）
    def fit_irls(self, X, y, max_iter=100, tol=1e-12):
        """用牛顿-拉夫森(IRLS)解同一个带 L2 的 MLE，用来校验梯度下降是否收敛到最优"""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        Xs = self.scaler_.transform(X) if self.standardize else X
        Xa = self._add_bias(Xs)
        w = np.zeros(Xa.shape[1])
        for it in range(max_iter):
            p = sigmoid(Xa @ w)                                   # 当前概率
            g = (Xa.T @ (p - y)) / Xa.shape[0]                    # 梯度
            g[:-1] += self.l2 * w[:-1]
            W = p * (1.0 - p) + 1e-12                             # 二阶导中的权重
            H = ((Xa * W[:, None]).T @ Xa) / Xa.shape[0]          # Hessian 主项
            H[np.arange(Xa.shape[1] - 1), np.arange(Xa.shape[1] - 1)] += self.l2
            step = np.linalg.solve(H, g)                          # 牛顿方向
            w = w - step                                          # 牛顿更新
            if np.max(np.abs(step)) < tol:                        # 牛顿步足够小则停止
                break
        J, _, _ = self._loss_grad_at(Xa, y, w)
        return w, it + 1, J

    def _loss_grad_at(self, Xa, y, w):
        """给定权重向量时的损失与梯度（供 IRLS 校验用）"""
        save = self.w_
        self.w_ = w
        out = self._loss_grad(Xa, y)
        self.w_ = save
        return out


print("LogisticRegressionGD 定义完成，方法列表：",
      [m for m in dir(LogisticRegressionGD) if not m.startswith("__")])
'''))

CELLS.append(("md", r"""
### 2.5 超参数选择：L2 系数用训练集内的 5 折交叉验证确定

**为什么必须有 L2？** 本数据集在 30 维空间里近似线性可分：存在一个超平面把
训练集几乎完全分开。此时不加正则的极大似然解**不存在**——损失可以无限接近 0，
但最优权重会发散到无穷。下面用实验直接验证：λ=0 时梯度范数一直降不下去，
牛顿法迭代轮数也异常。

**λ 怎么选？** 只在**训练集内部**做 5 折交叉验证，用验证折的平均对数损失选 λ，
**测试集一次都不碰**。这样选出来的超参数才有说服力。
"""))

CELLS.append(("code", r'''def cv_logloss(X_tr, y_tr, l2, k=5, seed=0):
    """训练集内部的 k 折交叉验证：返回各验证折平均对数损失的均值"""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y_tr))          # 打乱索引
    folds = np.array_split(idx, k)            # 切成 k 份
    scores = []
    for i in range(k):
        te_idx = folds[i]                                  # 第 i 折作验证
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])  # 其余作训练
        m = LogisticRegressionGD(l2=l2, lr_mode="lipschitz", tol=1e-10, max_iter=30000)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        p = m.predict_proba(X_tr[te_idx])[:, 1]            # 验证折的正类概率
        p = np.clip(p, 1e-15, 1 - 1e-15)                   # 防止 log(0)
        scores.append(-np.mean(y_tr[te_idx] * np.log(p) + (1 - y_tr[te_idx]) * np.log(1 - p)))
    return float(np.mean(scores))


lambda_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
rows = []
for lam in lambda_grid:
    m = LogisticRegressionGD(l2=lam, lr_mode="lipschitz", tol=1e-10, max_iter=30000).fit(X_train, y_train)
    w_irls, irls_iters, J_irls = m.fit_irls(X_train, y_train)        # 牛顿法解同一个问题
    rows.append({
        "lambda": lam,
        "CV_logloss": cv_logloss(X_train, y_train, lam),             # 训练集内 5 折 CV
        "GD_iters": m.n_iter_,
        "stop": m.stop_reason_,
        "train_loss": m.loss_hist_[-1],
        "grad_inf": m.grad_norm_hist_[-1],
        "test_acc": float(np.mean(m.predict(X_test) == y_test)),
        "IRLS_loss": J_irls,
        "w_gap": float(np.max(np.abs(w_irls - m.w_))),
        "w_norm": float(np.linalg.norm(m.w_[:-1])),
    })

lambda_table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("表 2-1  逻辑回归 L2 系数的选择（CV 只在训练集内进行）")
print(lambda_table.round(6).to_string(index=False))
print()
best_lam = float(lambda_table.loc[lambda_table["CV_logloss"].idxmin(), "lambda"])
print("交叉验证对数损失最小: lambda = %g  （CV logloss = %.5f）" % (best_lam, lambda_table["CV_logloss"].min()))

# 用最优 lambda 重新训练最终模型
LR_FINAL = LogisticRegressionGD(l2=best_lam, lr_mode="lipschitz", tol=1e-10, max_iter=30000)
t0 = time.time()
LR_FINAL.fit(X_train, y_train)
print("最终模型训练完成：%d 轮，用时 %.2f 秒" % (LR_FINAL.n_iter_, time.time() - t0))
print("  停止原因          : %s" % LR_FINAL.stop_reason_)
print("  Lipschitz 常数 L  : %.4f，理论步长 eta=1/L = %.4f" % (LR_FINAL.L_, LR_FINAL.eta_))
print("  初始/最终训练损失 : %.6f -> %.6f" % (LR_FINAL.loss_hist_[0], LR_FINAL.loss_hist_[-1]))
print("  最终梯度无穷范数  : %.2e" % LR_FINAL.grad_norm_hist_[-1])
print("  权重 L2 范数      : %.4f" % np.linalg.norm(LR_FINAL.w_[:-1]))
'''))

CELLS.append(("md", r"""
上表的解读：

- **λ = 0 与 λ = 1e-4 跑满 30000 轮都没收敛**（stop 列显示 max_iter），梯度范数停在 1e-4 量级，
  权重范数还在不断变大——这正是"近似线性可分时 MLE 不存在"的直接证据。
- 加上合适的 L2（λ = 1e-2）后，一千多轮就收敛，且梯度下降的权重与牛顿法（IRLS）
  解出的权重差值只有 1e-3 量级，说明梯度下降确实到达了同一个最优解。
- λ 过大（1.0）时正则项主导，训练损失明显变大，测试精度也下降，属于**欠拟合**；
  λ 过小则收敛慢、系数大，属于**过拟合**倾向。CV 选出的 λ 落在两者之间。
"""))

CELLS.append(("code", r'''# 把 lambda 选择过程画出来：CV 对数损失 与 权重范数
fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
pos = np.arange(len(lambda_grid))

ax1.plot(pos, lambda_table["CV_logloss"], "o-", color="#C0392B", lw=2, ms=7,
         label="5 折 CV 对数损失（越小越好）")
ax1.set_xlabel(r"L2 正则系数 $\lambda$")
ax1.set_ylabel("CV 对数损失", color="#C0392B")
ax1.tick_params(axis="y", labelcolor="#C0392B")
ax1.set_xticks(pos)
ax1.set_xticklabels([str(v) for v in lambda_grid])
best_pos = lambda_grid.index(best_lam)
ax1.axvline(best_pos, color="#7F8C8D", ls="--", lw=1.2)
ax1.annotate(r"$\lambda^*$=" + f"{best_lam:g}",
             xy=(best_pos, lambda_table["CV_logloss"].min()),
             xytext=(12, 22), textcoords="offset points", color="#7F8C8D", fontsize=11)

ax2 = ax1.twinx()
ax2.plot(pos, lambda_table["w_norm"], "s--", color="#2E86C1", lw=2, ms=6,
         label="权重 L2 范数")
ax2.set_ylabel("权重 L2 范数", color="#2E86C1")
ax2.tick_params(axis="y", labelcolor="#2E86C1")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_lambda_grid.png"))
plt.show()

print("λ 越大权重范数越小；CV 损失先降后升，最小值在 λ = %g" % best_lam)
'''))

CELLS.append(("md", r"""
### 2.6 学习率与特征标准化对梯度下降的影响（消融实验）

这一节回答任务提示里的问题："**注意特征尺度对逻辑回归梯度下降的影响（建议先标准化）**"。

做法：分别用**原始尺度**与**标准化后**的数据训练同一个模型，对比

1. 梯度 Lipschitz 常数 L 与理论上界步长 1/L（相差多少个数量级）；
2. 特征矩阵的条件数（决定收敛快慢）；
3. 相同迭代轮数下的损失曲线与梯度范数曲线；
4. 不同学习率下的行为（步长过大会振荡甚至发散）。
"""))

CELLS.append(("code", r'''# ---------- 1) 尺度条件对比 ----------
lr_std = LogisticRegressionGD(l2=1e-2, lr_mode="lipschitz", tol=1e-10, max_iter=30000).fit(X_train, y_train)
lr_raw = LogisticRegressionGD(l2=1e-2, lr_mode="lipschitz", tol=1e-10, max_iter=30000,
                              standardize=False).fit(X_train, y_train)

Xa_std = lr_std._add_bias(lr_std.scaler_.transform(X_train))
Xa_raw = lr_raw._add_bias(X_train)
cond_std = np.linalg.cond((Xa_std.T @ Xa_std) / len(y_train))
cond_raw = np.linalg.cond((Xa_raw.T @ Xa_raw) / len(y_train))

print("表 2-2  标准化前后的数值条件对比")
print("%-26s%16s%18s%14s" % ("项目", "标准化后", "原始尺度", "倍数"))
print("%-26s%16.4f%18.1f%14.0f" % ("梯度 Lipschitz 常数 L", lr_std.L_, lr_raw.L_, lr_raw.L_ / lr_std.L_))
print("%-26s%16.4f%18.3e%14.0f" % ("理论上界步长 eta = 1/L", lr_std.eta_, lr_raw.eta_, lr_std.eta_ / lr_raw.eta_))
print("%-26s%16.1f%18.3e%14.1e" % ("X^T X/N 的条件数", cond_std, cond_raw, cond_raw / cond_std))
print("%-26s%16.6f%18.6f%14s" % ("30000 轮后训练损失", lr_std.loss_hist_[-1], lr_raw.loss_hist_[-1], ""))
print("%-26s%16s%18s%14s" % ("是否收敛", str(lr_std.converged_), str(lr_raw.converged_), ""))
print("%-26s%16.4f%18.4f%14s" % ("测试集准确率",
                                np.mean(lr_std.predict(X_test) == y_test),
                                np.mean(lr_raw.predict(X_test) == y_test), ""))
'''))

CELLS.append(("code", r'''# ---------- 2) 标准化后：不同学习率的损失曲线 ----------
lr_values = [0.05, 0.1, 0.2, 1.0 / lr_std.L_, 0.5, 1.0]
colors = ["#8E44AD", "#2E86C1", "#27AE60", "#117A65", "#E67E22", "#C0392B"]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
for lr_val, c in zip(lr_values, colors):
    m = LogisticRegressionGD(lr=lr_val, lr_mode="fixed", l2=1e-2, tol=1e-12,
                             max_iter=4000).fit(X_train, y_train)
    tag = "（=1/L）" if abs(lr_val - 1 / lr_std.L_) < 1e-9 else ""
    ax.plot(np.arange(1, len(m.loss_hist_) + 1), m.loss_hist_, lw=1.8, color=c,
            label=r"$\eta$=" + f"{lr_val:.4f}" + tag)
ax.set_xlabel("迭代轮数 t")
ax.set_ylabel("训练损失 J(w)")
ax.legend(fontsize=9)
ax.set_ylim(0.09, 0.75)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_lr_lr_grid.png"))
plt.show()

print("学习率略小于 1/L 时单调下降；接近 1/L 时下降最快；超过 1/L 后出现振荡")
'''))

CELLS.append(("code", r'''# ---------- 3) 标准化 vs 未标准化：收敛对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

axes[0].plot(np.arange(1, len(lr_std.loss_hist_) + 1), lr_std.loss_hist_,
             color="#27AE60", lw=2, label="已标准化（$\\eta$=1/L=%.3f）" % lr_std.eta_)
axes[0].plot(np.arange(1, len(lr_raw.loss_hist_) + 1), lr_raw.loss_hist_,
             color="#C0392B", lw=2, label="未标准化（$\\eta$=1/L=%.2e）" % lr_raw.eta_)
axes[0].set_xlabel("迭代轮数 t")
axes[0].set_ylabel("训练损失 J(w)")
axes[0].set_title("(a) 各自按理论上界步长 1/L", fontsize=11)
axes[0].legend(fontsize=8)

axes[1].semilogy(np.arange(1, len(lr_std.grad_norm_hist_) + 1), lr_std.grad_norm_hist_,
                 color="#27AE60", lw=2, label="已标准化")
axes[1].semilogy(np.arange(1, len(lr_raw.grad_norm_hist_) + 1), lr_raw.grad_norm_hist_,
                 color="#C0392B", lw=2, label="未标准化")
axes[1].axhline(1e-6, color="#7F8C8D", ls="--", lw=1.2, label="收敛判据 1e-6")
axes[1].set_xlabel("迭代轮数 t")
axes[1].set_ylabel("梯度无穷范数（对数轴）")
axes[1].set_title("(b) 梯度范数的下降速度", fontsize=11)
axes[1].legend(fontsize=8)

# 对数轴的刻度默认用 mathtext 排版，会因字体缺 "−" 号报警；改成纯文本格式
from matplotlib.ticker import FuncFormatter, NullFormatter
axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%.0e" % v))
axes[1].yaxis.set_minor_formatter(NullFormatter())

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_standardization.png"))
plt.show()

print("标准化后 %d 轮损失收敛到 %.6f；未标准化 30000 轮后损失仍为 %.4f"
      % (lr_std.n_iter_, lr_std.loss_hist_[-1], lr_raw.loss_hist_[-1]))
print("两者所需步长相差 %.0f 倍，这就是必须先标准化的原因" % (lr_std.eta_ / lr_raw.eta_))
'''))
