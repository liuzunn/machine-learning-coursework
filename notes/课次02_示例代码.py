# -*- coding: utf-8 -*-
"""课次02 · Python 语言基础 —— 课堂示例代码

用法：
    python 课次02_示例代码.py          # 依次运行全部示例
    python 课次02_示例代码.py 3         # 只运行第 3 节

对应 PPT《课次02_Python语言基础_v2.pptx》，教师演示用。
每个小节都用 print 分隔，便于在课堂上逐段运行、逐段讲解。
"""

import sys


def line(title):
    print("\n" + "=" * 56)
    print(title)
    print("=" * 56)


# ---------------------------------------------------------------- 1 变量是标签
def demo_reference():
    """第 7 页：变量是"标签"而非"盒子"；第 9 页：赋值与引用"""
    line("1. 变量是标签，不是盒子（引用语义）")

    a = [1, 2, 3]
    b = a                      # 不是复制，而是再贴一张标签
    print("a =", a, " id(a) =", id(a))
    print("b =", b, " id(b) =", id(b))
    print("a 与 b 指向同一对象吗？", a is b)

    b.append(4)                # 通过 b 修改
    print("\n执行 b.append(4) 之后：")
    print("a =", a, "  <- a 也变了")
    print("len(a) =", len(a), "  <- 答案是 4，不是 3")

    print("\n想要真正的副本？用切片：")
    c = a[:]                   # 切片产生新列表
    c.append(5)
    print("c = a[:]; c.append(5)")
    print("a =", a, "  <- a 不受影响")
    print("c =", c)


# ---------------------------------------------------------------- 2 可变与不可变
def demo_mutable():
    """第 8 页：可变类型与不可变类型"""
    line("2. 可变类型与不可变类型")

    print("--- 不可变：数值（印刷品）---")
    x = 3
    print("x =", x, " id(x) =", id(x))
    x = x + 1                  # 不是"改"，而是新建一个对象
    print("x = x + 1 之后：x =", x, " id(x) =", id(x), " <- 地址变了")

    print("\n--- 可变：列表（白板）---")
    lst = [1, 2, 3]
    print("lst =", lst, " id(lst) =", id(lst))
    lst.append(4)              # 原地修改
    print("lst.append(4) 之后：lst =", lst, " id(lst) =", id(lst), " <- 地址没变")

    print("\n--- 对函数传参的影响 ---")
    def f_number(n):
        n = n + 1
        print("  函数内 n =", n)

    def f_list(l):
        l[0] = 99
        print("  函数内 l =", l)

    num = 3
    arr = [1, 2, 3]
    f_number(num)
    print("调用后 num =", num, " <- 不变（安全）")
    f_list(arr)
    print("调用后 arr =", arr, " <- 变了（危险！）")

    print("\n保护原列表：传入副本 f_list(arr[:])")
    arr2 = [1, 2, 3]
    f_list(arr2[:])
    print("调用后 arr2 =", arr2, " <- 原列表未受影响")


# ---------------------------------------------------------------- 3 运算符
def demo_operator():
    """第 14、16-19 页：运算符与优先级、短路求值"""
    line("3. 运算符")

    print("--- 加号的多重含义 ---")
    print("1 + 2        =", 1 + 2)
    print("[1, 2] + [3] =", [1, 2] + [3])
    print("'abc' + 'de' =", "abc" + "de")

    print("\n--- 乘号的多重含义 ---")
    print("3 * 4        =", 3 * 4)
    print("[1, 2] * 2   =", [1, 2] * 2)
    print("'哈' * 3     =", "哈" * 3)
    print("True * 3     =", True * 3, " <- True 参与运算时等于 1")
    print("True + 3     =", True + 3)

    print("\n--- 链式比较（Python 特有）---")
    print("1 < 3 < 5    =", 1 < 3 < 5)
    print("1 < 5 < 3    =", 1 < 5 < 3)

    print("\n--- 短路求值 ---")
    print("3 < 5 or 1/0 > 0 =", 3 < 5 or 1 / 0 > 0)
    print("  ^ 前者为真，后者根本不执行，所以不会除零报错")
    print("3 > 5 and 1/0 > 0 =", 3 > 5 and 1 / 0 > 0)
    print("  ^ 前者为假，后者同样不执行")

    print("\n--- 优先级：记不住就加括号 ---")
    print("2 + 3 * 4        =", 2 + 3 * 4)
    print("(2 + 3) * 4      =", (2 + 3) * 4)


# ---------------------------------------------------------------- 4 控制结构
def demo_control():
    """第 21-23 页：if / while / for"""
    line("4. 控制结构")

    print("--- if / elif / else（缩进就是代码块）---")
    score = 85
    if score >= 90:
        level = "优秀"
    elif score >= 60:
        level = "及格"
    else:
        level = "不及格"
    print("score =", score, " -> ", level)

    print("\n--- 三元运算符 ---")
    level2 = "及格" if score >= 60 else "不及格"
    print("level2 =", level2)

    print("\n--- for 遍历 ---")
    for name in ["甲", "乙", "丙"]:
        print("  同学：", name)

    print("\n--- range(n) 计数 ---")
    for i in range(3):
        print("  第", i, "次")

    print("\n--- while 条件循环 ---")
    n = 3
    while n > 0:
        print("  倒计时：", n)
        n -= 1

    print("\n--- 循环 + 判断：统计及格人数 ---")
    scores = [85, 92, 78, 55, 60]
    count = 0
    for s in scores:
        if s >= 60:
            count = count + 1
    print("scores =", scores)
    print("及格人数 =", count)


# ---------------------------------------------------------------- 5 函数
def demo_function():
    """第 28-30 页：函数定义、参数传递、作用域"""
    line("5. 函数")

    def count_above(data, threshold):
        """统计 data 中大于 threshold 的元素个数"""
        count = 0
        for x in data:
            if x > threshold:
                count += 1
        return count

    scores = [85, 92, 78, 55, 60]
    print("def count_above(data, threshold): ...")
    print("count_above(scores, 60) =", count_above(scores, 60))
    print("count_above(scores, 80) =", count_above(scores, 80))

    print("\n--- 作用域：UnboundLocalError ---")
    print("下面这段代码会报错（已捕获，仅作演示）：")
    print("    x = 3")
    print("    def f():")
    print("        print(x)   # 报错")
    print("        x = 5")
    try:
        code = "x = 3\ndef f():\n    print(x)\n    x = 5\nf()"
        exec(code)
    except Exception as e:
        print("  实际报错 ->", type(e).__name__, ":", e)
    print("  原因：函数内存在对 x 的赋值，Python 便将其视为局部变量，")
    print("        因此在赋值前引用会报错。")
    print("  解决：① 使用 global 声明；② 更推荐——用 return 返回结果。")


# ---------------------------------------------------------------- 6 模块导入
def demo_import():
    """第 31 页：import 的三种写法"""
    line("6. 模块导入")

    import math
    print("import math          -> math.sqrt(16) =", math.sqrt(16))

    import math as m
    print("import math as m     -> m.sqrt(16)   =", m.sqrt(16))

    from math import sqrt
    print("from math import sqrt-> sqrt(16)     =", sqrt(16))

    print("\n配套《Python 之禅》：在解释器里输入 import this")


# ---------------------------------------------------------------- 7 切片
def demo_slice():
    """第 34-35 页：切片语法与两个特例"""
    line("7. 切片 [start:end:step]  —— 左闭右开")

    a = [10, 20, 30, 40, 50]
    print("a =", a)
    print("a[1:4]   =", a[1:4], "  <- 下标 1、2、3，不包含 4")
    print("a[:3]    =", a[:3], "  <- 从头开始")
    print("a[2:]    =", a[2:], "  <- 到末尾")
    print("a[::2]   =", a[::2], "  <- 步长 2，隔一个取一个")
    print("a[::-1]  =", a[::-1], "  <- 步长 -1，逆序")
    print("a[10:20] =", a[10:20], "  <- 越界不报错，返回空列表")

    print("\n切片的 end 永远取不到 —— 这是本节最需要记住的一条规则")


# ---------------------------------------------------------------- 8 案例：循环次数
def demo_case():
    """第 24-25 页：GTA Online 案例 —— 循环次数对性能的影响"""
    line("8. 案例：循环的执行次数（GTA Online 加载问题）")

    import time

    n = 63000
    comparisons = (n * n + n) // 2
    print(f"商品条目数 n      = {n:,}")
    print(f"逐项比对所需次数  = {comparisons:,} 次（约 {comparisons / 1e8:.1f} 亿）")
    print("实际结果：加载时间由 6 分钟降至 1 分 50 秒（减少约 70%）")

    print("\n在同一台机器上对比两种做法（用较小的 n 演示）：")
    n = 6000

    # 做法一：每加入一项，都遍历整个列表检查是否重复（O(n^2)）
    start = time.time()
    seen = []
    for i in range(n):
        found = False
        for j in seen:            # 全表扫描
            if j == i:
                found = True
                break
        if not found:
            seen.append(i)
    t1 = time.time() - start

    # 做法二：用集合（哈希表）检查是否重复（O(n)）
    start = time.time()
    seen2 = set()
    for i in range(n):
        if i not in seen2:        # 哈希查找
            seen2.add(i)
    t2 = time.time() - start

    print(f"  双重循环（数组扫描）：{t1:.4f} 秒")
    print(f"  单循环 + 集合查重  ：{t2:.4f} 秒")
    print(f"  耗时比            ：{t1 / t2 if t2 > 0 else float('inf'):.1f} 倍")
    print("\n结论：循环体多执行一次，开销就多一分；数据结构的选择同样决定执行次数。")

    print("\n延伸：预计算（打表法）—— 能提前算的，提前算好")
    start = time.time()
    table = [i * i for i in range(n)]      # 事先算好
    for i in range(n):
        _ = table[i]                       # 用时直接查表
    t3 = time.time() - start
    print(f"  预计算 + 查表       ：{t3:.4f} 秒（含建表时间）")
    print("  适用条件：输入范围有限且可预先确定，以空间换时间。")


# ---------------------------------------------------------------- 主程序
SECTIONS = [
    ("1", "变量是标签（引用）", demo_reference),
    ("2", "可变与不可变", demo_mutable),
    ("3", "运算符", demo_operator),
    ("4", "控制结构", demo_control),
    ("5", "函数与作用域", demo_function),
    ("6", "模块导入", demo_import),
    ("7", "切片", demo_slice),
    ("8", "案例：循环次数", demo_case),
]

if __name__ == "__main__":
    print("课次02 · Python 语言基础 —— 课堂示例代码")
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg:
        for key, name, func in SECTIONS:
            if key == arg:
                func()
                break
        else:
            print("未找到该小节。可选：", " / ".join(k for k, _, _ in SECTIONS))
    else:
        for key, name, func in SECTIONS:
            func()
        print("\n" + "=" * 56)
        print("全部示例运行完毕。")
