# 独立合成器交叉核验（pyLIQTR / pygridsynth）

**目的**：回答审稿人最可能提的那个问题——"你的 T 门数标签凭什么可信"。
阶段一原有 5 项自检（GHZ 全 Clifford、ε 单调、T 深度序关系、qiskit 独立基展开、ε 斜率 3.05）
全部是**内部自洽**，无法排除"我们自己算错了"。本文档记录外部实现的对账结果。

## 为什么不能在 Mac 上做

`pyLIQTR`（含 `pygridsynth`）依赖 `numba → llvmlite`。macOS + CPython 3.12 没有预编译 wheel，
pip 会退化成源码编译 LLVM，在 `.venv-liqtr` 里稳定失败（`CalledProcessError ... ffi/build.py`）。
Linux 上一条 `pip install pyLIQTR pygridsynth` 就过。**这一环与机器算力无关，只与操作系统有关。**

## 远端环境（一次性，可复现）

```
# 机器：AutoDL 实例，128 核 / 503 GB / RTX 4080 SUPER，Python 3.10
conda create -y -n liqtr python=3.10
/root/miniconda3/envs/liqtr/bin/pip install pyLIQTR pygridsynth
```

## 实验设计

- **控制变量**：同一批 (θ, ε) 上双端独立合成，比较 T 门数与实测角度误差。
- **角度集**：500 个 θ = 300 个均匀随机一般角 + QFT/QPE 型 `kπ/2^m` 角（含非 Clifford 项），
  精度档 ε ∈ {1e-3, 1e-6, 1e-9} → **1,500 组**；固定种子，可复现。
- **左端（本仓库生产实现）**：`qiskit.synthesis.discrete_basis.ross_selinger.gridsynth_rz`（Rust 后端）。
  实测其输出电路的门集只有 `{h, s, t, x}`——没有 `tdg/sdg`，所以"数 `t` 的个数"就是 T 数，不存在口径歧义。
- **右端（独立实现）**：`pygridsynth.gridsynth_gates`（Ross 原始代码谱系的 Python + mpmath 实现）。
  其 `W` 门是全局相位门 `exp(iπ/8)·I`，不计 T，故 T 数 = 门串中 `T` 的个数。

## 结果

| ε | T 中位（本仓库） | T 中位（独立实现） | 完全相等 | 比值 ≤1.10 | 比值中位 | p10–p90 |
|---|---|---|---|---|---|---|
| 1e-9 | 92 | 94 | 45.6% | **99.4%** | 1.011 | 1.000–1.033 |
| 1e-6 | 62 | 64 | 48.8% | **99.0%** | 1.000 | 1.000–1.049 |
| 1e-3 | 32 | 32 | 47.0% | 87.8% | 1.000 | 1.000–1.103 |

**精度契约（比"数字一样"更硬的一条）**：把本仓库的合成结果逐条重建酉矩阵量实际角度误差，
`err/ε` 中位 **0.356**、p90 **0.746**，**0/1500 越界**；pygridsynth 侧同样 100% ≤ ε（最大 9.96e-10 @ ε=1e-9）。
这说明标签不是"为了 T 数好看而放宽精度"，每个旋转真的被压进了声明的 ε 预算。

**每 log2(1/ε) 的 T 代价**：3.077（1e-9）/ 3.111（1e-6）/ 3.211（1e-3），理论值 3——
与 §4.6 的 ε 斜率自检（中位 3.05）互相印证：两条独立路径给出同一个常数。

## 两个"不一致"的解释（不是 bug）

1. **1e-3 档约 12% 的点比值 >1.10**：粗精度下大量 θ 满足 `|θ − kπ/4| < ε`，qiskit 会吸附成 1 个 T 的
   Clifford 等价，pygridsynth 仍走一般路径给出约 30 个 T。两端都满足 ε 约束，差异来自"是否利用 ε 松弛"。
2. 近 Clifford 角（`kπ/4 + 1e-7`）在 1e-6 档出现 1 vs 80 T：同上，属实现策略差而非错误。

## 结论与残留局限

- 实现级一致：中位 T 数与逐点分布都吻合（1e-6/1e-9 档 10% 内 ≥99%），**不存在系统性高估或低估**；
- 精度契约实测成立（0 越界），这是标签可信度的直接证据；
- **仍缺算法级独立**：两端同属 Ross–Selinger 算法族，验证的是实现不是最优性。
  算法级需要 Qualtran / Microsoft QRE，均未接入，论文中应如实写为 limitation。

## 复现命令

```bash
# 1) 本机：生成角度集 + 本仓库 T 数 + ε 合规审计
.venv/bin/python scripts/xcheck_export.py          # -> xcheck_in.csv / qiskit_eps_audit.csv
# 2) 远端：独立实现
scp xcheck_in.csv <host>:/root/qbench/in.csv
python scripts/xcheck_remote_runner.py /root/qbench/in.csv /root/qbench/out.csv
# 3) 回传 out.csv 后本机合并
.venv/bin/python scripts/xcheck_merge.py           # -> xcheck_pairs.csv / xcheck_summary.json
```
