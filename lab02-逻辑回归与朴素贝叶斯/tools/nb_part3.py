# -*- coding: utf-8 -*-
"""Notebook 第 3 部分：手写高斯朴素贝叶斯"""

CELLS = []

CELLS.append(("md", r"""
## 3. 高斯朴素贝叶斯的手写实现

### 3.1 算法原理

#### （1）贝叶斯决策

对样本 `\mathbf{x}`，要判断它属于哪一类，按**最大后验概率**（MAP）决策：

$$ \hat{y} = \arg\max_{c \in \{0,1\}} P(y = c \mid \mathbf{x}) $$

由贝叶斯公式

$$ P(y = c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y = c)\, P(y = c)}{P(\mathbf{x})} $$

分母 `P(\mathbf{x})` 与类别无关，比较时可以直接忽略，所以只需比较**分子**。

#### （2）条件独立假设（"朴素"二字的来源）

`P(\mathbf{x} \mid y = c)` 是 30 维的联合密度，直接估计需要天文数字的样本量。
朴素贝叶斯引入**条件独立假设**：在给定类别 c 的条件下，30 个特征相互独立，

$$ P(\mathbf{x} \mid y = c) = \prod_{j=1}^{d} P(x_j \mid y = c) $$

这把一个 30 维密度估计问题，化简成 30 个一维密度估计问题。
**这个假设是本方法全部误差的来源**：若特征间存在强相关（1.4 节已经看到大量 |r|>0.9 的特征对），
独立假设被违背，模型会重复计算同一份信息。

#### （3）高斯类条件密度

假设每个特征在每个类别下服从正态分布：

$$ P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi\sigma_{cj}^{2}}}
\exp\left( -\frac{(x_j - \mu_{cj})^{2}}{2\sigma_{cj}^{2}} \right) $$

参数用**极大似然**闭式估计（就是组内样本均值与组内样本方差）：

$$ \hat{\mu}_{cj} = \frac{1}{N_c}\sum_{i:\,y_i=c} x_{ij}, \qquad
\hat{\sigma}_{cj}^{2} = \frac{1}{N_c}\sum_{i:\,y_i=c} (x_{ij} - \hat{\mu}_{cj})^{2} $$

先验同样用频率估计：`\hat{P}(y=c) = N_c / N`。

#### （4）对数域计算

把三式代回并取对数（连乘变连加，避免 30 个小于 1 的数相乘下溢）：

$$ \ln P(y=c \mid \mathbf{x}) = \ln P(y=c) + \sum_{j=1}^{d}\left[
-\tfrac{1}{2}\ln(2\pi\sigma_{cj}^{2}) - \frac{(x_j - \mu_{cj})^{2}}{2\sigma_{cj}^{2}} \right] - \ln P(\mathbf{x}) $$

其中最后一项对所有类别相同，可省略；省略后得到的是**未归一化的对数后验** `s_c`。
需要输出概率时，对 `\{s_c\}` 做 **log-sum-exp 归一化**：

$$ P(y=c \mid \mathbf{x}) = \frac{\exp(s_c - \max_{c'} s_{c'})}
{\sum_{c'} \exp(s_{c'} - \max_{c''} s_{c''})} $$

先减去最大值是纯数值技巧：它不改变结果（分子分母同乘一个常数），却能保证 `\exp` 不会上溢。

#### （5）数值稳定性：方差下限保护

若某个特征在某一类里是常数，`\hat{\sigma}_{cj}^{2} = 0`，公式会依次出现

1. `\ln(2\pi\sigma^{2}) \to -\infty`；
2. `(x_j - \mu_{cj})^{2}/(2\sigma^{2})` 出现 **除以零**；
3. 若恰好 `x_j = \mu_{cj}`，得到 **0/0 = NaN**。

任意一种都会让整条样本的后验概率变成 `nan`，进而使全部预测失效。
解决办法是给方差设一个下限：

$$ \tilde{\sigma}_{cj}^{2} = \max\left(\hat{\sigma}_{cj}^{2},\ \varepsilon\right), \qquad \varepsilon = 10^{-9} $$

取这么小的 `\varepsilon` 有两个理由：一方面远小于数据本身的量纲，不会影响正常特征的估计
（本数据最小类条件方差约 4e-6，比 1e-9 大三个数量级）；另一方面足以避免除零与对数发散。
3.4 节会用"人为加入一列常数列"的方式把这个保护机制的效果演示出来。
"""))

CELLS.append(("md", r"""
### 3.2 手写高斯朴素贝叶斯类

代码结构：

| 方法 | 作用 |
|---|---|
| `fit` | 统计类先验、类条件均值与方差，做方差下限保护，预计算对数参数 |
| `joint_log_likelihood` | 计算 (N,2) 的对数联合概率，二次型展开成两次矩阵乘法 |
| `predict_log_proba` | 用 log-sum-exp 归一化得到对数后验 |
| `predict_proba` | 取指数得到后验概率 |
| `predict` | 取对数联合概率最大的类别 |

注意：朴素贝叶斯**没有迭代求解过程**，参数全部是闭式估计，训练一次就结束。
"""))

CELLS.append(("code", r'''class GaussianNBFromScratch:
    """高斯朴素贝叶斯：类先验 + 每类每特征的高斯类条件密度，全程对数域计算"""

    def __init__(self, var_floor=1e-9, use_floor=True, ddof=0):
        self.var_floor = var_floor   # 方差下限 epsilon（数值保护）
        self.use_floor = use_floor   # 是否启用下限保护（关闭后用于演示 nan）
        self.ddof = ddof             # 0=极大似然方差；1=无偏方差

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)                 # 类别集合 {0, 1}
        n, d = X.shape                               # 样本数与特征数
        K = len(self.classes_)
        self.prior_ = np.zeros(K)                    # 类先验 P(y=c)
        self.mean_ = np.zeros((K, d))                # 类条件均值 mu_cj
        self.var_ = np.zeros((K, d))                 # 类条件方差 sigma^2_cj
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]                           # 取出第 c 类的全部样本
            self.prior_[k] = Xc.shape[0] / n         # 先验 = 该类样本占比
            self.mean_[k] = Xc.mean(axis=0)          # 组内均值（极大似然估计）
            self.var_[k] = Xc.var(axis=0, ddof=self.ddof)  # 组内方差（极大似然估计）
        if self.use_floor:
            self.var_ = np.maximum(self.var_, self.var_floor)   # 方差下限保护
        self.log_prior_ = np.log(self.prior_)        # 对数先验
        self.log_norm_ = -0.5 * np.log(2.0 * np.pi * self.var_)   # -0.5*ln(2*pi*sigma^2)
        self.inv_var_ = 1.0 / self.var_              # 1/sigma^2，后面反复用
        return self

    def joint_log_likelihood(self, X):
        """(N,K) 的对数联合概率 ln P(x, y=c)，二次型展开成两次矩阵乘法"""
        X = np.asarray(X, float)
        # ln P(x,y=c) = ln P(c) + sum_j[ -0.5ln(2*pi*sigma^2) - (x_j-mu)^2/(2*sigma^2) ]
        # 展开平方项： -x^2/(2sigma^2) + x*mu/sigma^2 - mu^2/(2sigma^2)
        quad_x = -0.5 * (X ** 2) @ self.inv_var_.T                    # -sum x_j^2/(2 sigma^2)
        lin_x = X @ (self.mean_ * self.inv_var_).T                    # +sum x_j mu/sigma^2
        const = (-0.5 * np.sum(self.mean_ ** 2 * self.inv_var_, axis=1)  # -sum mu^2/(2 sigma^2)
                 + np.sum(self.log_norm_, axis=1))                    # -0.5 sum ln(2 pi sigma^2)
        return quad_x + lin_x + const + self.log_prior_                # 再加对数先验

    def predict_log_proba(self, X):
        """用 log-sum-exp 归一化，得到 (N,K) 的对数后验"""
        jll = self.joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)   # 减去 ln P(x) 的估计

    def predict_proba(self, X):
        """取指数得到后验概率（数值上已由 log-sum-exp 保证不溢出）"""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """MAP 决策：取对数联合概率最大的类别"""
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]


print("GaussianNBFromScratch 定义完成，方法列表：",
      [m for m in dir(GaussianNBFromScratch) if not m.startswith("__")])
'''))

CELLS.append(("md", r"""
### 3.3 参数估计结果

训练只做一次闭式估计，没有迭代。下面检查估计出来的先验、均值、方差是否合理，
并确认方差下限保护在真实数据上"没有改变任何结果"（因为真实最小方差远大于 ε）。
"""))

CELLS.append(("code", r'''# 主实验：直接在原始特征上训练（朴素贝叶斯对特征尺度不敏感，3.5 节会验证）
GNB = GaussianNBFromScratch(var_floor=1e-9, ddof=0).fit(X_train, y_train)

print("类先验 P(y=c)：")
for k, c in enumerate(GNB.classes_):
    print("  P(y=%d) = %.6f   （训练集中 %d / %d 条）"
          % (c, GNB.prior_[k], np.sum(y_train == c), len(y_train)))

print("\n类条件均值的若干示例（原始量纲）：")
print("%-22s%14s%14s%14s" % ("特征", "mu(恶性)", "mu(良性)", "差值"))
for name in ["mean radius", "mean texture", "mean concavity", "worst area", "worst concave points"]:
    j = list(feature_names).index(name)
    print("%-22s%14.4f%14.4f%14.4f"
          % (name, GNB.mean_[0, j], GNB.mean_[1, j], GNB.mean_[1, j] - GNB.mean_[0, j]))

print("\n类条件方差的分布（原始量纲）：")
print("  全部 %d 个 (类别,特征) 组合中：最小值 = %.3e，最大值 = %.3e"
      % (GNB.var_.size, GNB.var_.min(), GNB.var_.max()))
print("  小于方差下限 1e-9 的组合个数 :", int(np.sum(GNB.var_ <= 1e-9)))
print("  小于 1e-6 的组合个数         :", int(np.sum(GNB.var_ < 1e-6)))
print("  结论：真实数据上没有退化方差，下限保护不改变任何估计值")

# 验证：开启/关闭下限保护的结果应完全一致（因为真实方差都大于下限）
GNB_nofloor = GaussianNBFromScratch(var_floor=1e-9, use_floor=False).fit(X_train, y_train)
p_a = GNB.predict_proba(X_test)
p_b = GNB_nofloor.predict_proba(X_test)
print("\n启用 vs 关闭下限保护（真实数据）：概率最大绝对差 = %.3e，预测完全一致 = %s"
      % (np.max(np.abs(p_a - p_b)), bool(np.all(GNB.predict(X_test) == GNB_nofloor.predict(X_test)))))
'''))

CELLS.append(("md", r"""
### 3.4 方差下限保护的必要性（对照实验）

构造一个**退化特征**：在训练集和测试集末尾各加一列常数 0。
这一列在每个类别内的方差都恰好为 0，用来模拟"某个指标在某一类中恒定不变"的极端情况
（真实数据里，若某一类样本很少、某特征又恰好相同，就会出现这种情形）。

对比两种情况：

- **不启用保护**：`\ln(2\pi\cdot 0) = -\infty`、除以 0，后验出现 nan，整个预测失效；
- **启用保护**（`\varepsilon=10^{-9}`）：后验有限，预测正常。
"""))

CELLS.append(("code", r'''# 人工构造一列类内方差恒为 0 的退化特征
X_train_degen = np.hstack([X_train, np.zeros((len(X_train), 1))])
X_test_degen = np.hstack([X_test, np.zeros((len(X_test), 1))])

with np.errstate(divide="ignore", invalid="ignore"):    # 这里就是要复现"除零"，屏蔽警告
    gnb_bad = GaussianNBFromScratch(use_floor=False).fit(X_train_degen, y_train)   # 不启用保护
gnb_ok = GaussianNBFromScratch(var_floor=1e-9, use_floor=True).fit(X_train_degen, y_train)  # 启用保护

print("退化特征的类条件方差（最后一行最后一列）：")
print("  不启用保护 :", gnb_bad.var_[:, -1])
print("  启用保护   :", gnb_ok.var_[:, -1])

with np.errstate(divide="ignore", invalid="ignore"):
    jll_bad = gnb_bad.joint_log_likelihood(X_test_degen[:5])
    jll_ok = gnb_ok.joint_log_likelihood(X_test_degen[:5])
print("\n前 5 个测试样本的对数联合概率（不启用保护）：")
print(np.round(jll_bad, 4))
print("前 5 个测试样本的对数联合概率（启用保护）：")
print(np.round(jll_ok, 4))

with np.errstate(divide="ignore", invalid="ignore"):
    proba_bad = gnb_bad.predict_proba(X_test_degen)
proba_ok = gnb_ok.predict_proba(X_test_degen)
print("\n不启用保护时后验含 nan :", bool(np.isnan(proba_bad).any()))
print("启用保护时后验含 nan   :", bool(np.isnan(proba_ok).any()))
print("不启用保护时的准确率   : 无法计算（预测全部失效）")
print("启用保护时的准确率     : %.4f" % np.mean(gnb_ok.predict(X_test_degen) == y_test))

assert np.isnan(proba_bad).any() and not np.isnan(proba_ok).any()
print("\n断言通过：不加下限保护必然产生 nan，加上之后完全正常 —— 这就是方差下限保护的必要性")
'''))

CELLS.append(("md", r"""
### 3.5 尺度不变性：标准化对高斯朴素贝叶斯没有影响

这是一个很值得写进报告的性质。把每个特征做仿射变换 `x_j \to a_j x_j + b_j`，
高斯类条件密度中的均值与标准差同步变换 `\mu \to a\mu+b`、`\sigma \to |a|\sigma`，
代入对数密度后，每个特征**只多出一个与类别无关的常数** `\ln|a_j|`。
由于 MAP 决策只比较不同类别之间的相对大小，这些常数会被完全抵消。

**理论结论**：对每个特征单独做线性变换（标准化、归一化等），
高斯朴素贝叶斯的预测**完全不变**（但取对数属于非线性变换，会改变结果）。

这和逻辑回归形成鲜明对比：逻辑回归必须先标准化（2.6 节），高斯朴素贝叶斯则无所谓。
下面用实验直接验证。
"""))

CELLS.append(("code", r'''# 在标准化后的数据上重新训练朴素贝叶斯
sc_gnb = StandardScalerFromScratch().fit(X_train)
GNB_std = GaussianNBFromScratch(var_floor=1e-9, ddof=0).fit(sc_gnb.transform(X_train), y_train)

p_raw = GNB.predict_proba(X_test)                        # 原始尺度下的后验
p_std = GNB_std.predict_proba(sc_gnb.transform(X_test))  # 标准化后的后验

print("理论预期：两者的预测与后验概率应完全一致")
print("  预测标签完全一致   :", bool(np.all(GNB.predict(X_test) == GNB_std.predict(sc_gnb.transform(X_test)))))
print("  后验概率最大绝对差 : %.3e" % np.max(np.abs(p_raw - p_std)))
print("  准确率（原始尺度） : %.4f" % np.mean(GNB.predict(X_test) == y_test))
print("  准确率（标准化）   : %.4f" % np.mean(GNB_std.predict(sc_gnb.transform(X_test)) == y_test))

assert np.allclose(p_raw, p_std, atol=1e-10), "尺度不变性被破坏"
print("\n断言通过：高斯朴素贝叶斯对逐特征线性变换具有严格的不变性")
print("=> 报告中不必为它做标准化；但若做「特征取对数」这类非线性变换，结果会变")
'''))
