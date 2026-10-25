# -*- coding: utf-8 -*-
"""生成两张中文算法流程图（逻辑回归 / 高斯朴素贝叶斯）。

只写入 figures/fig1_lr_flowchart.png 与 figures/fig2_gnb_flowchart.png。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

# ---------------------------------------------------------------------------
# 统一风格
# ---------------------------------------------------------------------------
COLOR_START = "#E8F0FE"
COLOR_PROCESS = "#FFFFFF"
COLOR_DECISION = "#FFF4D6"
COLOR_IO = "#E9F7EF"
COLOR_EDGE_BLUE = "#3C6E9E"
COLOR_EDGE = "#444444"
COLOR_ARROW = "#333333"
COLOR_TEXT = "#20262E"
COLOR_MUTED = "#555555"
COLOR_TITLE = "#1F3F5C"

FS_TITLE = 13.0
FS_BODY = 10.8
FS_TERM = 11.6
FS_LABEL = 10.0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": COLOR_TEXT,
        "savefig.edgecolor": "white",
    }
)


def new_canvas():
    """画布宽 7 英寸；去掉图内标题后高度缩为 9.70 英寸，数据坐标按英寸记。"""
    fig = plt.figure(figsize=(7, 9.70), dpi=150)
    # 用一块整画布的白底保证 bbox_inches="tight" 时仍保留完整版面宽度。
    fig.add_artist(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="none",
            zorder=-100,
            clip_on=False,
        )
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7)
    # 顶部只留 0.20 英寸 padding，第一个节点不会贴边。
    ax.set_ylim(0, 9.70)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def title(ax, text):
    t = ax.text(
        3.5,
        9.78,
        text,
        ha="center",
        va="center",
        fontsize=FS_TITLE,
        fontweight="bold",
        color=COLOR_TITLE,
        zorder=6,
    )
    # SimHei 没有粗体字重，轻描边让标题更醒目、更接近加粗效果。
    t.set_path_effects([pe.withStroke(linewidth=0.28, foreground=COLOR_TITLE)])
    return t


def rounded_box(
    ax,
    cx,
    cy,
    w,
    h,
    text,
    *,
    kind="process",
    fontsize=FS_BODY,
    dashed=False,
    text_ha="center",
    text_x=None,
):
    if kind == "process":
        face, edge = COLOR_PROCESS, COLOR_EDGE
    elif kind == "io":
        face, edge = COLOR_IO, COLOR_EDGE_BLUE
    elif kind == "loop":
        face, edge = COLOR_PROCESS, COLOR_EDGE_BLUE
    else:
        raise ValueError(kind)

    pad_eff = 0.075
    # 需求指定 boxstyle="round,pad=0.3,rounding_size=0.15"。
    # 通过 mutation_scale 把 0.3 的数据单位换算成约 0.075 英寸。
    box = FancyBboxPatch(
        (cx - w / 2 + pad_eff, cy - h / 2 + pad_eff),
        w - 2 * pad_eff,
        h - 2 * pad_eff,
        boxstyle="round,pad=0.3,rounding_size=0.15",
        mutation_scale=pad_eff / 0.3,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
        linestyle=(0, (5, 2)) if dashed else "solid",
        zorder=2,
    )
    ax.add_patch(box)
    txt = ax.text(
        text_x if text_x is not None else cx,
        cy,
        text,
        ha=text_ha,
        va="center",
        fontsize=fontsize,
        color=COLOR_TEXT,
        linespacing=1.35,
        zorder=3,
    )
    return box, txt


def terminal(ax, cx, cy, w, h, text):
    """起止节点：椭圆。"""
    patch = Ellipse(
        (cx, cy),
        w,
        h,
        facecolor=COLOR_START,
        edgecolor=COLOR_EDGE_BLUE,
        linewidth=1.3,
        zorder=2,
    )
    ax.add_patch(patch)
    txt = ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=FS_TERM,
        color=COLOR_TEXT,
        zorder=3,
    )
    return patch, txt


def diamond(ax, cx, cy, w, h, text):
    pts = [
        (cx, cy + h / 2),
        (cx + w / 2, cy),
        (cx, cy - h / 2),
        (cx - w / 2, cy),
    ]
    patch = Polygon(
        pts,
        closed=True,
        facecolor=COLOR_DECISION,
        edgecolor=COLOR_EDGE,
        linewidth=1.3,
        zorder=2,
    )
    ax.add_patch(patch)
    txt = ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=FS_BODY,
        color=COLOR_TEXT,
        linespacing=1.35,
        zorder=3,
    )
    return patch, txt


def arrow(ax, a, b, *, scale=8.8, lw=1.4):
    patch = FancyArrowPatch(
        a,
        b,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=COLOR_ARROW,
        shrinkA=0,
        shrinkB=0,
        capstyle="round",
        joinstyle="round",
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def poly_arrow(ax, vertices, *, scale=8.8, lw=1.4):
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(vertices) - 1)
    patch = FancyArrowPatch(
        path=MPath(vertices, codes),
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=COLOR_ARROW,
        shrinkA=0,
        shrinkB=0,
        capstyle="round",
        joinstyle="round",
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def edge_label(ax, x, y, text, *, rotation=0):
    return ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=FS_LABEL,
        color=COLOR_MUTED,
        rotation=rotation,
        rotation_mode="anchor",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none"),
        zorder=5,
    )


def loop_icon(ax, cx, cy, r=0.13):
    arc = MPath.arc(20, 340)
    verts = arc.vertices.copy()
    verts[:, 0] = cx + r * verts[:, 0]
    verts[:, 1] = cy + r * verts[:, 1]
    patch = FancyArrowPatch(
        path=MPath(verts, arc.codes),
        arrowstyle="-|>",
        mutation_scale=7.2,
        linewidth=1.2,
        color=COLOR_EDGE_BLUE,
        shrinkA=0,
        shrinkB=0,
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def check_text_containment(fig, ax, entries, canvas_name):
    """保存前做一次几何自检：文字是否越出自己的框、是否越出画布。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    problems = []
    for item in entries:
        txt = item["text"]
        bb = txt.get_window_extent(renderer=renderer).transformed(inv)
        c = item.get("container")
        if c is not None:
            x0, y0, x1, y1 = c
            inset = 0.01
            if bb.x0 < x0 + inset or bb.x1 > x1 - inset or bb.y0 < y0 + inset or bb.y1 > y1 - inset:
                problems.append(
                    f"{item['name']}: text={bb.bounds} container={(x0, y0, x1, y1)}"
                )
        if bb.x0 < 0.02 or bb.x1 > 6.98 or bb.y0 < 0.02 or bb.y1 > 9.98:
            problems.append(f"{item['name']}: outside canvas {bb.bounds}")
    if problems:
        print(f"[{canvas_name}] 几何检查发现 {len(problems)} 个问题：")
        for p in problems:
            print("  -", p)
    else:
        print(f"[{canvas_name}] 几何检查通过：所有文字均在框内且未越出画布。")
    return problems


# ---------------------------------------------------------------------------
# 图 1：逻辑回归
# ---------------------------------------------------------------------------
def build_lr():
    fig, ax = new_canvas()
    entries = []

    x = 3.5
    W = 5.6

    terminal(ax, x, 9.31, 1.7, 0.38, "开始")

    input_top, input_h = 9.02, 0.66
    b, t = rounded_box(
        ax,
        x,
        input_top - input_h / 2,
        W,
        input_h,
        "输入：训练集 X（N×d）、标签 y∈{0,1}、学习率 η、\n"
        "最大迭代 T、容忍度 tol、L2 系数 λ",
        kind="io",
    )
    entries.append({"name": "输入", "text": t, "container": (x - W / 2, input_top - input_h, x + W / 2, input_top)})

    s1_top, s1_h = 8.26, 0.62
    b, t = rounded_box(
        ax,
        x,
        s1_top - s1_h / 2,
        W,
        s1_h,
        "步骤 1  对 X 每列估计均值 μ 与标准差 σ，\n"
        "并对 X 做 z-score 标准化",
    )
    entries.append({"name": "步骤1", "text": t, "container": (x - W / 2, s1_top - s1_h, x + W / 2, s1_top)})

    s2_top, s2_h = 7.54, 0.62
    b, t = rounded_box(
        ax,
        x,
        s2_top - s2_h / 2,
        W,
        s2_h,
        "步骤 2  构造增广矩阵 X_aug = [X, 1]，\n"
        "把偏置 b 并入权重向量 w",
    )
    entries.append({"name": "步骤2", "text": t, "container": (x - W / 2, s2_top - s2_h, x + W / 2, s2_top)})

    s3_top, s3_h = 6.82, 0.54
    b, t = rounded_box(
        ax,
        x,
        s3_top - s3_h / 2,
        W,
        s3_h,
        "步骤 3  初始化 w = 0（第 0 轮），J_old = +∞",
    )
    entries.append({"name": "步骤3", "text": t, "container": (x - W / 2, s3_top - s3_h, x + W / 2, s3_top)})

    s4_top, s4_h = 6.18, 0.62
    b, t = rounded_box(
        ax,
        x,
        s4_top - s4_h / 2,
        W,
        s4_h,
        "步骤 4  前向计算 z = X_aug w，\n"
        "概率 p = σ(z) = 1 / (1 + e^(-z))",
    )
    entries.append({"name": "步骤4", "text": t, "container": (x - W / 2, s4_top - s4_h, x + W / 2, s4_top)})

    s5_top, s5_h = 5.46, 0.84
    b, t = rounded_box(
        ax,
        x,
        s5_top - s5_h / 2,
        W,
        s5_h,
        "步骤 5  计算负对数似然（交叉熵）：\n"
        "J(w) = -(1/N) Σ [ y ln p + (1-y) ln(1-p) ]\n"
        "+ (λ/2) ||w[1:d]||^2",
    )
    entries.append({"name": "步骤5", "text": t, "container": (x - W / 2, s5_top - s5_h, x + W / 2, s5_top)})

    s6_top, s6_h = 4.52, 0.76
    b, t = rounded_box(
        ax,
        x,
        s6_top - s6_h / 2,
        W,
        s6_h,
        "步骤 6  计算梯度 g = (1/N) X_aug^T (p - y) + λw\n"
        "（偏置项 b 不加正则化）",
    )
    entries.append({"name": "步骤6", "text": t, "container": (x - W / 2, s6_top - s6_h, x + W / 2, s6_top)})

    s7_top, s7_h = 3.66, 0.52
    b, t = rounded_box(
        ax,
        x,
        s7_top - s7_h / 2,
        W,
        s7_h,
        "步骤 7  更新参数 w ← w - η g",
    )
    entries.append({"name": "步骤7", "text": t, "container": (x - W / 2, s7_top - s7_h, x + W / 2, s7_top)})

    dcy = 2.32
    dw, dh = 6.0, 1.44
    b, t = diamond(
        ax,
        x,
        dcy,
        dw,
        dh,
        "收敛判定：|J_new - J_old| < tol\n"
        "或 ||g||_∞ < tol，或达到最大迭代 T ？",
    )
    entries.append({"name": "判定", "text": t, "container": (x - dw / 2, dcy - dh / 2, x + dw / 2, dcy + dh / 2)})

    s8_top, s8_h = 1.42, 0.64
    b, t = rounded_box(
        ax,
        x,
        s8_top - s8_h / 2,
        W,
        s8_h,
        "步骤 8  输出最优参数 w 与训练损失曲线 J(t)",
        kind="io",
    )
    entries.append({"name": "步骤8", "text": t, "container": (x - W / 2, s8_top - s8_h, x + W / 2, s8_top)})

    terminal(ax, x, 0.51, 1.7, 0.38, "结束")

    # 主流程箭头
    arrow(ax, (x, 9.12), (x, 9.02))
    arrow(ax, (x, 8.36), (x, 8.26))
    arrow(ax, (x, 7.64), (x, 7.54))
    arrow(ax, (x, 6.92), (x, 6.82))
    arrow(ax, (x, 6.28), (x, 6.18))
    arrow(ax, (x, 5.56), (x, 5.46))
    arrow(ax, (x, 4.62), (x, 4.52))
    arrow(ax, (x, 3.76), (x, 3.66))
    arrow(ax, (x, 3.14), (x, 3.04))
    arrow(ax, (x, 1.60), (x, 1.42))
    arrow(ax, (x, 0.78), (x, 0.70))

    # “否”分支：从菱形左顶点绕到左侧，再折回步骤 4 左边框。
    y4 = 5.87
    poly_arrow(
        ax,
        [(0.50, dcy), (0.22, dcy), (0.22, y4), (0.70, y4)],
        scale=8.8,
    )
    edge_label(ax, 0.80, 2.44, "否")
    edge_label(ax, 0.87, 2.02, "J_old ← J_new")
    edge_label(ax, 0.445, 4.08, "返回步骤 4", rotation=90)
    edge_label(ax, 3.72, 1.51, "是")

    problems = check_text_containment(fig, ax, entries, "图 1")
    return fig, problems


# ---------------------------------------------------------------------------
# 图 2：高斯朴素贝叶斯
# ---------------------------------------------------------------------------
def build_gnb():
    fig, ax = new_canvas()
    entries = []

    x = 3.5
    W = 5.6

    terminal(ax, x, 9.32, 1.7, 0.36, "开始")

    input_top, input_h = 9.04, 0.64
    b, t = rounded_box(
        ax,
        x,
        input_top - input_h / 2,
        W,
        input_h,
        "输入：训练集 X、标签 y、方差下限保护项 ε\n"
        "（平滑/保护项，防止除零与对数发散）",
        kind="io",
    )
    entries.append({"name": "输入", "text": t, "container": (x - W / 2, input_top - input_h, x + W / 2, input_top)})

    s1_top, s1_h = 8.30, 0.64
    b, t = rounded_box(
        ax,
        x,
        s1_top - s1_h / 2,
        W,
        s1_h,
        "步骤 1  统计类别集合 C 与每类样本数 N_c，\n"
        "先验概率 P(y=c) = N_c / N",
    )
    entries.append({"name": "步骤1", "text": t, "container": (x - W / 2, s1_top - s1_h, x + W / 2, s1_top)})

    s2_top, s2_h = 7.56, 0.64
    b, t = rounded_box(
        ax,
        x,
        s2_top - s2_h / 2,
        W,
        s2_h,
        "步骤 2  对每个类别 c∈C、每个特征 j=1..d，\n"
        "估计均值 μ_cj 与方差 σ_cj^2",
    )
    entries.append({"name": "步骤2", "text": t, "container": (x - W / 2, s2_top - s2_h, x + W / 2, s2_top)})

    s3_top, s3_h = 6.82, 0.54
    b, t = rounded_box(
        ax,
        x,
        s3_top - s3_h / 2,
        W,
        s3_h,
        "步骤 3  方差下限保护：σ_cj^2 ← max(σ_cj^2, ε)",
    )
    entries.append({"name": "步骤3", "text": t, "container": (x - W / 2, s3_top - s3_h, x + W / 2, s3_top)})

    s4_top, s4_h = 6.18, 0.64
    b, t = rounded_box(
        ax,
        x,
        s4_top - s4_h / 2,
        W,
        s4_h,
        "步骤 4  预计算对数先验 ln P(y=c)，\n"
        "以及每类对数似然参数 -0.5 ln(2πσ_cj^2)",
    )
    entries.append({"name": "步骤4", "text": t, "container": (x - W / 2, s4_top - s4_h, x + W / 2, s4_top)})

    loop_top, loop_h = 5.44, 0.50
    loop_cy = loop_top - loop_h / 2
    rounded_box(
        ax,
        x,
        loop_cy,
        W,
        loop_h,
        "循环：对测试样本 x 逐个预测",
        kind="loop",
        dashed=True,
        text_ha="left",
        text_x=2.56,
    )
    loop_icon(ax, 2.28, loop_cy, r=0.13)

    s5_top, s5_h = 4.84, 0.86
    b, t = rounded_box(
        ax,
        x,
        s5_top - s5_h / 2,
        W,
        s5_h,
        "步骤 5  对每个类别 c 计算对数后验（忽略常数项）：\n"
        "s_c = ln P(y=c) + Σ_j [ -0.5 ln(2πσ_cj^2)\n"
        "- (x_j - μ_cj)^2 / (2σ_cj^2) ]",
    )
    entries.append({"name": "步骤5", "text": t, "container": (x - W / 2, s5_top - s5_h, x + W / 2, s5_top)})

    s6_top, s6_h = 3.88, 0.80
    b, t = rounded_box(
        ax,
        x,
        s6_top - s6_h / 2,
        W,
        s6_h,
        "步骤 6  用 log-sum-exp 归一化：\n"
        "P(y=c|x) = exp(s_c - max s) /\n"
        "Σ_c' exp(s_c' - max s)",
    )
    entries.append({"name": "步骤6", "text": t, "container": (x - W / 2, s6_top - s6_h, x + W / 2, s6_top)})

    s7_top, s7_h = 2.98, 0.68
    b, t = rounded_box(
        ax,
        x,
        s7_top - s7_h / 2,
        W,
        s7_h,
        "步骤 7  取概率最大的类别作为预测：\n"
        "y_hat = argmax_c P(y=c|x)，并输出该样本的后验概率",
    )
    entries.append({"name": "步骤7", "text": t, "container": (x - W / 2, s7_top - s7_h, x + W / 2, s7_top)})

    dcy = 1.81
    dw, dh = 5.6, 0.78
    b, t = diamond(
        ax,
        x,
        dcy,
        dw,
        dh,
        "所有测试样本\n都已处理完？",
    )
    entries.append({"name": "判定", "text": t, "container": (x - dw / 2, dcy - dh / 2, x + dw / 2, dcy + dh / 2)})

    out_top, out_h = 1.22, 0.56
    b, t = rounded_box(
        ax,
        x,
        out_top - out_h / 2,
        W,
        out_h,
        "输出预测标签 y_hat 与正类概率 P(y=1|x)",
        kind="io",
    )
    entries.append({"name": "输出", "text": t, "container": (x - W / 2, out_top - out_h, x + W / 2, out_top)})

    terminal(ax, x, 0.40, 1.7, 0.36, "结束")

    # 主流程箭头
    arrow(ax, (x, 9.14), (x, 9.04))
    arrow(ax, (x, 8.40), (x, 8.30))
    arrow(ax, (x, 7.66), (x, 7.56))
    arrow(ax, (x, 6.92), (x, 6.82))
    arrow(ax, (x, 6.28), (x, 6.18))
    arrow(ax, (x, 5.54), (x, 5.44))
    arrow(ax, (x, 4.94), (x, 4.84))
    arrow(ax, (x, 3.98), (x, 3.88))
    arrow(ax, (x, 3.08), (x, 2.98))
    arrow(ax, (x, 2.30), (x, 2.20))
    arrow(ax, (x, 1.42), (x, 1.22))
    arrow(ax, (x, 0.66), (x, 0.58))

    # “否”分支：从菱形左顶点绕到左侧，再折回步骤 5 左边框。
    y5 = 4.41
    poly_arrow(
        ax,
        [(0.70, dcy), (0.22, dcy), (0.22, y5), (0.70, y5)],
        scale=8.8,
    )
    edge_label(ax, 0.82, 1.93, "否")
    edge_label(ax, 0.445, 3.05, "返回步骤 5", rotation=90)
    edge_label(ax, 3.72, 1.32, "是")

    problems = check_text_containment(fig, ax, entries, "图 2")
    return fig, problems


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        edgecolor="white",
    )
    plt.close(fig)


def main():
    fig1, p1 = build_lr()
    fig2, p2 = build_gnb()
    save(fig1, FIG_DIR / "fig1_lr_flowchart.png")
    save(fig2, FIG_DIR / "fig2_gnb_flowchart.png")
    print("已生成：")
    print(" ", (FIG_DIR / "fig1_lr_flowchart.png").resolve())
    print(" ", (FIG_DIR / "fig2_gnb_flowchart.png").resolve())
    if p1 or p2:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
