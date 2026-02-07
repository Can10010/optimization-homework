# 实验五：NSGA-III 求解多目标优化问题

## 0. 代码来源

老师给的 NSGA-III 算法代码与以下开源项目一致（或高度一致）：

* [作者仓库](https://github.com/Xavier-MaYiMing/NSGA-III)

原仓库作者在代码头部也写了算法参考文献（Deb & Jain, 2014）。
本实验**不上传/不直接搬运**老师下发的整套代码文件，仅在日志中说明“如何把原始 `NSGA_III.py` 改造成可测试多个标准测试函数的版本”。

---

## 1. 这个实验到底要测什么

题目要求：“测试 NSGA-III 算法，给出不少于 4 个测试函数的测试结果”。

这里的**测试函数**不是“Problem 文件夹里那一堆 py”，而是指多目标优化领域的**标准基准函数（benchmark）**，常见就是：

* **ZDT 系列**：ZDT1 / ZDT2 / ZDT3 / ZDT4 / ZDT6（固定 2 目标，容易画图）
* **DTLZ 系列**：DTLZ1 / 2 / 3 / 4 / 7（可设目标数 m，可画 2 或 3 目标）

**老师给的一堆 Problem 文件**很多是框架或拓展用途，不是“完成作业必须跑的”。
这次实验只要：**给 NSGA-III 一个“目标函数计算接口”**，然后选 benchmark 跑起来即可。

---

## 2. 为什么我不输出 CSV（只画图）

因为这类实验的核心是 **Pareto 前沿的形状与覆盖情况**：

* 2 目标：二维散点图最直观
* 3 目标：三维散点图还能看
* 多 目标：光靠图很难看，需要 HV/IGD 等指标（题目并没要求）

所以本实验只保留图像结果（可保存 png），不导出 csv。

---

## 3. 最重要：从原版 NSGA_III.py 改成“可换测试函数”的版本（手把手）

### 3.1 原版哪里“写死了 DTLZ1”

在原版 `NSGA_III.py` 里，有一个：

```python
def cal_obj(pop, nobj):
    # DTLZ1
    ...
```

这意味着：**无论你想测什么，它都只会算 DTLZ1**。
所以要改的关键就是：

> 把 `cal_obj(pop, nobj)` 改成 `cal_obj(pop, problem_callable)`
> 并且在 main 开头创建 `problem_callable`（根据 PROBLEM_NAME 选择 ZDT/DTLZ）

---

### 3.2 第一步：在文件顶部加一个“配置区”（直接复制粘贴）

把下面这段加到 import 后面（建议放在最上面）：

```python
# =========================
# 你只需要改这里（配置区）
# =========================
PROBLEM_NAME = "ZDT1"     # 改成：ZDT1/ZDT2/ZDT3/ZDT4/ZDT6/DTLZ1/DTLZ2/DTLZ3/DTLZ4/DTLZ7
M_OBJECTIVES = 2          # DTLZ 用：目标数 m（2 或 3 方便画图）；ZDT 固定 2
N_VARIABLES = 30          # 决策变量数 n：ZDT 常用 30；DTLZ 常用 7/12/30 都可
NPOP = 120
N_ITER = 300
SEED = 1

SAVE_FIG = True           # 保存图片
SHOW_FIG = True           # 是否弹窗显示
```

---

### 3.3 第二步：把“写死 DTLZ1 的 cal_obj”替换掉（整段替换）

**找到原来的 `def cal_obj(pop, nobj):`，整段删掉，用下面替换：**

```python
def cal_obj(pop, problem_callable):
    """
    pop: (npop, nvar)
    problem_callable: 一个可调用对象，输入一个个体向量 -> 输出 [f1,f2,...]
    """
    objs = np.asarray([problem_callable(ind) for ind in pop], dtype=float)
    if objs.ndim == 1:
        objs = objs[:, None]
    return objs
```

---

### 3.4 第三步：新增一个“按名称创建测试函数”的函数（直接复制粘贴）

在文件中随便找个位置（建议放在 `main()` 前面）加入：

```python
def make_problem(name: str, m: int, n: int):
    """
    返回: (problem_callable, lb, ub)
    problem_callable(ind) -> objectives
    """
    name = name.upper().strip()
    from optproblems import zdt, dtlz

    # ===== ZDT（固定 2 目标）=====
    if name == "ZDT1":
        p = zdt.ZDT1(num_variables=n)
        return p, np.zeros(n), np.ones(n)
    if name == "ZDT2":
        p = zdt.ZDT2(num_variables=n)
        return p, np.zeros(n), np.ones(n)
    if name == "ZDT3":
        p = zdt.ZDT3(num_variables=n)
        return p, np.zeros(n), np.ones(n)
    if name == "ZDT4":
        p = zdt.ZDT4(num_variables=n)
        lb = np.array([0] + [-5] * (n - 1), dtype=float)
        ub = np.array([1] + [5] * (n - 1), dtype=float)
        return p, lb, ub
    if name == "ZDT6":
        p = zdt.ZDT6(num_variables=n)
        return p, np.zeros(n), np.ones(n)

    # ===== DTLZ（目标数 m 可设）=====
    if name == "DTLZ1":
        p = dtlz.DTLZ1(m, n)
        return p, np.zeros(n), np.ones(n)
    if name == "DTLZ2":
        p = dtlz.DTLZ2(m, n)
        return p, np.zeros(n), np.ones(n)
    if name == "DTLZ3":
        p = dtlz.DTLZ3(m, n)
        return p, np.zeros(n), np.ones(n)
    if name == "DTLZ4":
        p = dtlz.DTLZ4(m, n)
        return p, np.zeros(n), np.ones(n)
    if name == "DTLZ7":
        p = dtlz.DTLZ7(m, n)
        return p, np.zeros(n), np.ones(n)

    raise ValueError(f"不支持的 PROBLEM_NAME: {name}")
```

---

### 3.5 第四步：改 main() 的输入，让它吃“problem_callable”

**原版 main 是：**

```python
def main(npop, iter, lb, ub, nobj=3, pc=1, pm=1, eta_c=30, eta_m=20):
```

你把它改成（只改函数签名和内部两处 cal_obj 调用）：

```python
def main(npop, n_iter, lb, ub, problem_callable, pc=1, pm=1, eta_c=30, eta_m=20):
    nvar = len(lb)
    pop = np.random.uniform(lb, ub, (npop, nvar))

    objs = cal_obj(pop, problem_callable)
    nobj = objs.shape[1]

    # 只画 2 或 3 目标
    if nobj > 3:
        print(f"[SKIP] {PROBLEM_NAME}: nobj={nobj} (>3) 不画图，跳过。")
        return None, nobj

    V = reference_points(npop, nobj)
    zmin = np.min(objs, axis=0)
    _, rank = nd_sort(objs)

    for t in range(n_iter):
        if (t + 1) % 50 == 0:
            print(f"Iteration {t+1}/{n_iter} completed.")

        mating_pool = selection(pop, pc, rank)
        off = crossover(mating_pool, lb, ub, pc, eta_c)
        off = mutation(off, lb, ub, pm, eta_m)
        off_objs = cal_obj(off, problem_callable)

        zmin = np.minimum(zmin, np.min(off_objs, axis=0))
        pop, objs, rank = environmental_selection(
            np.concatenate((pop, off), axis=0),
            np.concatenate((objs, off_objs), axis=0),
            zmin, npop, V
        )

    pf = objs[rank == 0]
    return pf, nobj
```

---

### 3.6 第五步：把最后 `if __name__ == '__main__'` 改成“只改名字就能跑”

把原版最后三行：

```python
if __name__ == '__main__':
    main(91, 400, np.array([0] * 7), np.array([1] * 7))
```

替换为：

```python
def plot_pf(pf: np.ndarray, name: str):
    if pf is None:
        return
    nobj = pf.shape[1]
    if nobj == 2:
        plt.figure()
        plt.scatter(pf[:, 0], pf[:, 1], s=10)
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title(f"Pareto Front - {name}")
    elif nobj == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], s=10)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.set_title(f"Pareto Front - {name}")

    if SAVE_FIG:
        plt.savefig(f"pf_{name}.png", dpi=200, bbox_inches="tight")
        print(f"[SAVED] pf_{name}.png")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    if SEED is not None:
        np.random.seed(SEED)

    problem, lb, ub = make_problem(PROBLEM_NAME, M_OBJECTIVES, N_VARIABLES)
    pf, nobj = main(NPOP, N_ITER, lb, ub, problem)

    if pf is None:
        raise SystemExit(0)

    print(f"[DONE] {PROBLEM_NAME} | PF size={pf.shape[0]} | nobj={nobj}")
    plot_pf(pf, f"{PROBLEM_NAME}_m{M_OBJECTIVES}_n{N_VARIABLES}")
```

---

## 4. 参数怎么调

* `PROBLEM_NAME`：换测试函数
* `M_OBJECTIVES`：DTLZ 才需要（2 或 3 就行，便于画图）
* `N_VARIABLES`：变量维度
* `NPOP / N_ITER`：种群规模与迭代次数（越大越慢，但前沿更密）

---

## 5. 哪些算法/函数不建议测（一句话就够）

* 超过 3 目标的 many-objective 问题不适合用“散点图”展示，若要做需要额外指标（本实验不要求）。


