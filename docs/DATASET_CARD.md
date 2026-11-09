# QuantumAI-FTQC Dataset Card（v0.1-MVP）

> 状态：MVP 阶段数据集卡片。所有数字由 `scripts/build_dataset.py` / `qec_resources.py` / `verify_labels.py`
> 的实际输出统计得到（日志见 `logs/`，可用 `scripts/report_stats.py` 复算）。
> 本卡片**不**声称任何尚未完成的采集。QASMBench 与 QDataSet 本体仍未落地（见 §2），
> QCircuitBench 的 qasm 本体已在本机但尚未纳入标签流水线，这两种状态在 §2 中分开记录。

## 1. 数据集定位

`QuantumAI-FTQC` 是一个面向**容错量子计算（FTQC）资源估计**的三层结构化数据集：

| 层 | 表 | 记录数 | 单位 | 标签来源 |
|---|---|---|---|---|
| 线路层 | `processed_data/circuit_features.parquet` | **7,502** | 每条线路一行 | Qiskit 结构解析 |
| 逻辑资源层 | `processed_data/logical_resources.parquet` | **22,506** | 线路 × 3 档合成精度 ε | Ross–Selinger 逐角合成（Qiskit `gridsynth_rz`） |
| 物理 / QEC 层 | `processed_data/fault_tolerant_resources.parquet` | **202,554** | 线路 × 27 个配置 | surface code 解析参考估计器（本仓库 `scripts/qec_resources.py`） |

关键规模指标（必须同时报告，避免"样本量虚高"）：

| 指标 | 数值 |
|---|---|
| Unique circuit count | **7,502**（MQT Bench 1,246 + QCircuitBench 6,236） |
| Unique algorithm count | **192** |
| Unique configuration count | **27**（3 档 ε × 3 档物理错误率 × 3 档逻辑错误预算） |
| Total estimation records | **225,060**（22,506 逻辑 + 202,554 物理） |
| 线路量子比特范围 | 2 – 60 |
| 逻辑 T-count 范围 | 0 – **124,167,640**（其中 1,068 条全 Clifford，T-count ≡ 0；MQT 训练侧 131 条，占 10.5%） |
| 码距范围 | 3 – 29（物理比特 36 – 4.31e5，runtime 6e-6 – 378 s） |

采集清单 `raw_data/circuits_manifest.parquet` 共 **7,654** 条候选线路（`qasm_sha256` 无重复），
其中 **7,502** 条完成标注（已标注率 **98.0%**）；未标注的 **152** 条 = 61 条 ALG `n>20`（按 `ALG_LABEL_MAX_SIZE=20` 设计截断）
+ **74 条"中间测量 + `c_if` 经典反馈"线路**（qiskit 无法展开到酉基，T-count 依赖运行时测量结果、**无单一定义**，属概念上不可标注）
+ 17 条大 n INDEP（含 `grover n≥30` 永久卡死）。**注意**：早期版本把"大量 ALG 未标注"归因于 n>12 截断是错的，
真实主因是 qiskit 无法把带经典条件分支的线路导出为 OpenQASM 2（168 条），现已改用 **QPY** 补采（见 §5）。

## 2. 数据来源（provenance）

| 来源 | 抽象层次 | 采集方式 | 是否进入训练 |
|---|---|---|---|
| MQT Bench (`mqt.bench`) | `ALG`（黑盒算法算子） | `get_benchmark(algo, ALG, size, opt_level=0)` → OpenQASM2 / **QPY**（动态控制流） | 是（截断 n ≤ 20，见 §5） |
| MQT Bench | `INDEP`（硬件无关展开） | 同上；另用 `scripts/acquire_randomcircuit.py` 补同族不同 rep 实例 402 条 | 是（全规模） |
| QASMBench | — | **未完成**：codeload tarball 在本网络多次截断（8.4–11.6 MB），`gzip -t` 失败 | 否 |
| QCircuitBench | — | **本体已在本机**：`raw_data/QCircuitBench/` 含 16,589 个 `.qasm`（OpenQASM 3.0，567 MB）。抽样 120 个：原生可解析 54.2%，递归内联 `include` 后 **94.2%**；剩余失败为缺失自定义门定义（`Oracle` / `mcx`）。**已接入流水线**：见下一行 | — |
| QCircuitBench | `EXT`（外部来源，保留原始规模） | `scripts/import_qcircuitbench.py`：递归内联 `include` → `qasm3.loads` → 展开到 `{rz,sx,x,cx}` → 写入 OpenQASM2 并追加 manifest。支持 `--workers`（并发 chunk）与 `--skip-processed`（断点续跑）。累计导入 6236 条（≤60 比特全库随机抽样 6000 条，成功 5757、失败 243，成功率 **95.95%**；失败原因：`QASM2ExportError` 94、chunk 超时 40、缺自定义门 `Oracle`/`CU_0`/`mcx` 共 109） | **仅作跨源测试集**，不进入 A/B/C 训练 |
| QDataSet | — | 仅克隆代码仓库（README/示例），14 TB 数据本体未下载 | 否 |

因此 v0.2 的训练集仍是**单源**（MQT Bench，**1,246** 条：INDEP 787 / ALG 459，其中 randomcircuit 同族实例 397 条），但**已具备大规模跨源测试集**（QCircuitBench 6236 条已标注，占库内 16,589 条的 37.6%）。
这一点很关键：跨源泛化现在不是"无法证明"，而是"已在一个 6236 条线路的独立分布上测出有限的迁移能力"（见 `docs/STAGE1_REPORT.md` §4.4）。
注意：**跨源 MdAPE 随测试集组成变化显著**（956 条版 20.9% → 6236 条版 11.3%），任何跨版本比较必须同时报告测试集规模与组成。

## 3. 标签口径（论文必须逐字保持一致）

### 3.1 逻辑层

```
t_count(ε) = t_count_clifford_part + Σ_i T_cost(θ_i, ε)
```

- 先把线路展开到 `{rz, sx, x, cx}` 基，逐 `rz` 判别：
  - 角度是 π/4 的整数倍（容差 1e-8）→ Clifford 角，`k mod 8` 为偶数（I/S/Z/S†）记 **0** 个 T，为奇数（T/T†）记 **1** 个 T；
  - 非 Clifford 角 → `gridsynth_rz(θ, ε).count_ops()["t"]`（Ross–Selinger 精确合成）。
- `logical_qubits = num_qubits`（定义恒等，不作为可学习目标时的特征屏蔽见 §4）。
- `t_depth_structural`：每个旋转占 1 个 T 层（结构深度下界）；`t_depth_serial`：单比特串行上界。
  两者分开报，不混用。
- 三档合成精度：ε ∈ {1e-3, 1e-6, 1e-9}（`synthesis_epsilon` 列）。

### 3.2 物理 / QEC 层

解析参考模型（**不是** Microsoft QRE 读数，论文中不得混称）：

```
p_L   = 0.03 · (p / 1e-2)^((d+1)/2)          # surface code 逻辑错误率
rounds = logical_t_depth + circuit_depth
physical_qubits = 2·d²·(logical_qubits + F·15)   # F = magic state factory 数量
runtime = rounds · d · 1 µs
d 取满足 rounds · p_L(d) ≤ budget 的最小奇数码距（D_MAX = 41）
```

参数扫描：p ∈ {1e-4, 5e-4, 1e-3}，budget ∈ {1e-3, 1e-6, 1e-9}；`status` 列记录 `feasible` / 超预算。

**已知简化**（必须在论文 limitation 中写明）：
1. runtime 只计入码周期 × 轮数，未计入 magic state 工厂的吞吐与 T 门注入排队；
2. 单一 surface code 方案，未比较 color code / LDPC；
3. 工厂数量固定启发式（不是 Pareto 搜索），因此**没有**给出"最少比特 / 最短时间"两个端点。

## 4. 特征集

`STRUCT_FEATURES`（28 项）：`num_qubits`、`gate_count`、`circuit_depth`、单/双/多比特门数、`cx_count`、
原始 `t/tdg/ccx/rz` 计数、`gate_type_entropy`、`gate_density`、`gates_per_layer`、`depth_per_qubit`、
交互图统计（`interaction_edges`、`degree_mean/max`、`graph_density`、连通分量）、
Clifford+T 后结构量（`n_clifford_rz`、`n_nonclifford_rz`、`t_count_clifford_part`、`t_depth_structural`、
`clifford_ops_after_ct`、`cx_after_ct`）。

`CONFIG_FEATURES`（3 项）：`log_epsilon`、`log_p`、`log_budget`。
`CAT_FEATURES`：`algorithm`（仅 grouped 协议使用；LOSO / scale 协议下按定义不可用）。

**防泄漏约定**：预测 `logical_qubits` 时屏蔽 `num_qubits`（否则是恒等映射，R² 恒为 1、MAE ≈ 0，
会得出"模型完美"的假结论）。

## 5. 已知采集限制

| 限制 | 影响 | 处理 |
|---|---|---|
| ALG 层 n > 20 的 Clifford+T 展开规模爆炸（如 grover n=14 单条即 2.28 M T-count；n≥30 永久卡死、`SIGALRM` 无效） | 缺失超大规模 ALG 样本 | `ALG_LABEL_MAX_SIZE = 20` 截断（61 条）；17 条大 n INDEP 超时，记录在 `logs/label_timeouts.csv` |
| 动态控制流线路无法 `qasm2.dumps`（如 `dynamic_qft`） | 该批线路原无文件落盘 | **已解决**：改用 QPY 序列化（`scripts/acquire_dynamic.py`），加载统一走 `scripts/circuit_io.py`；QASM3 只能救回 2/4 族 |
| 中间测量 + `c_if` 经典反馈线路（74 条） | 报 `KeyError: <Clbit register=..., index=0>`，无法展开到酉基 | **不纳入数据集**：T-count 依赖运行时测量、无单一定义，属概念上不可标注；**放宽 timeout 无效**，已留档 |
| `qiskit` Rust transpiler 不响应 `SIGALRM` | 进程内超时不可靠 | 改为**分块子进程**标注（`label_worker.py` + `label_parallel.py`），超时 `proc.kill()` 硬杀 |
| xgboost 在 macOS 缺 `libomp`（`brew install libomp` 报 `unknown or unsupported macOS version: :dunno`；sklearn 自带 `libomp.dylib` 与之符号不兼容 `___kmpc_dispatch_deinit`） | XGBoost 不可用 | `train_models.py` 自动降级为 `HistGradientBoostingRegressor`（同类 GBDT），`XGB_BACKEND` 记录实际后端 |

## 6. 标签自检（`results/evaluation/label_checks.json`）

| 检查 | 结果 | 判据 |
|---|---|---|
| GHZ 全 Clifford 线路 T-count = 0 | passed | `t_count_clifford_part` 与 `n_nonclifford_rz` 均为 0 |
| T-count 对 ε 单调（ε 变小 T 不减） | passed | 0 / 7502 违例 |
| `t_depth_serial ≥ t_depth_structural` | passed | 0 违例 |
| 与 qiskit 独立基展开一致（ε=1e-10） | passed | max_rel_diff = 0.1874，median = 0.0924，qiskit 更低者 0/4 |

第 4 项的容差为 20%，原因是 qiskit 内建的 `rz → Clifford+T` 合成并非最优 gridsynth，两条路径的绝对 T 数
**本身不唯一**；方向性检查（`qiskit ≥ gridsynth`）全部成立，说明本流水线读数不偏高。

## 7. 使用与引用

```bash
.venv/bin/python scripts/report_stats.py        # 复算本卡片所有数字
.venv/bin/python scripts/verify_labels.py       # 复跑 4 项自检
.venv/bin/python scripts/train_models.py --models ridge rf xgb
```

数据许可：线路本体来自 MQT Bench（MIT / Apache-2.0 依上游仓库），本仓库生成的标签表按 CC-BY-4.0 释放；
引用上游时需同时引用 MQT Bench 论文（Quetschlich et al., *MQT Bench*, ACM TOG / arXiv:2303.12302）。
