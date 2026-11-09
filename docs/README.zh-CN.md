# QuantumAI-FTQC

AI-driven Fault-Tolerant Quantum Resource Estimation —— 一个可复现的实验框架：
从公开量子线路数据构建**容错量子计算（FTQC）资源数据集**，并用机器学习预测逻辑 / 物理资源。

> 本仓库当前处于 **MVP 阶段（阶段一）**：已生成"线路特征 + Clifford+T 逻辑资源 + surface code 物理资源"三层标签，
> 并完成四种数据划分协议下的基线模型实验（同源分组、留一算法、规模外推、跨源外推）。目录、脚本、口径均按论文可复现要求组织。
>
> **当前数据规模**：采集清单 **7654** 条 → 已标注 **7502** 条线路（**192** 个算法，**2–60** 比特）→ 逻辑标签 **22,506** 行（3 档 ε）
> → 物理/QEC 标签 **202,554** 行（27 个配置）。标签自检 4 项全通过（ε 单调性 0/7502 违例），
> 物理不变式检查：T-count 对 log2(1/ε) 的每角斜率中位 **3.03**（理论 3log2），
> 并有 **1,500 组独立合成器对账**（pyLIQTR/pygridsynth，ε 合规 0 越界）。
> 其中 MQT Bench **1,246** 条用于训练（INDEP 787 / ALG 459，含 randomcircuit 同族实例 397 条），QCircuitBench **6,236** 条只作跨源测试集；
> 未标注 152 条 = 61 条 ALG `n>20` 设计内跳过 + 74 条"中间测量 + `c_if` 经典反馈"概念上不可标注 + 17 条大 n 超时。
>
> 详细文档：[`docs/DATASET_CARD.md`](docs/DATASET_CARD.md)（数据口径与来源）、
> [`docs/STAGE1_REPORT.md`](docs/STAGE1_REPORT.md)（阶段实验结果与缺口）、
> [`docs/REMOTE_MIGRATION.md`](docs/REMOTE_MIGRATION.md)（迁到远程 Linux 主机的清单）。

## 1. 快速开始

```bash
cd QuantumAI-Bench
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# ① 采集线路（MQT Bench，ALG + INDEP 两个抽象层次，写入 OpenQASM2）
.venv/bin/python scripts/build_dataset.py --levels ALG INDEP --random-reps 4

# ② 逻辑资源标注（Clifford+T 分解 + Ross–Selinger 角度合成，跨 3 档 ε）
.venv/bin/python scripts/build_dataset.py --resume --timeout 300

# ②b 动态控制流线路（QASM2 无法导出）改用 QPY 补采
.venv/bin/python scripts/acquire_dynamic.py

# ②c randomcircuit 同族新实例（不同 rep，成本极低、提升多样性）
.venv/bin/python scripts/acquire_randomcircuit.py --reps 1 20 --max-size 24

# ②d 接入外部跨源线路（QCircuitBench，OpenQASM 3.0 → 递归内联 include → OpenQASM2）
.venv/bin/python scripts/import_qcircuitbench.py --sample --limit 300 --max-qubits 40
.venv/bin/python scripts/label_parallel.py --chunk 5 --timeout 120 --workers 4

# ③ 物理 / QEC 资源标注（surface code 解析模型，参数扫描）
.venv/bin/python scripts/qec_resources.py

# ④ 训练与评估（grouped / LOSO / scale / crosssrc 四种协议 × ridge / rf / xgb）
#    macOS 上没有 libomp 时 xgb 分支自动降级为 HistGradientBoostingRegressor（XGB_BACKEND 记录实际后端）
.venv/bin/python scripts/train_models.py --models ridge rf xgb --protocols grouped loso scale crosssrc --tag v8

# ④b 消融（只动训练侧，测试集不变）
.venv/bin/python scripts/train_models.py --protocols crosssrc --targets logical_t_count physical_qubits \
    --drop-clifford-train --tag v8nc
.venv/bin/python scripts/train_models.py --protocols crosssrc --targets logical_t_count physical_qubits \
    --drop-algos-train ae dynamic_qft ghz_dynamic iqpe --tag v8ndyn

# ⑤ 标签自检（GHZ 全 Clifford、ε 单调性、深度关系、独立路径交叉核验）
.venv/bin/python scripts/verify_labels.py

# ⑥ 统计显著性（以 circuit_id 为簇的 Cluster Bootstrap + 配对 Wilcoxon + Holm）+ 种子方差
.venv/bin/python scripts/stats_tests.py && .venv/bin/python scripts/seed_variance.py

# ⑦ 论文图（fig1–6 + fig7）
.venv/bin/python scripts/make_figures.py && .venv/bin/python scripts/make_figure7.py
```

## 2. 目录

| 路径 | 内容 |
|---|---|
| `raw_data/circuits/` | 线路本体（每条一个 `.qasm`，动态控制流族为 `.qpy`；文件名即 `circuit_id`） |
| `raw_data/circuits_manifest.parquet` | 线路清单：来源、算法、抽象层次、请求规模、去重哈希、文件路径 |
| `processed_data/circuit_features.parquet` | 线路结构特征（37 列，含图统计与 Clifford+T 结构量） |
| `processed_data/logical_resources.parquet` | 逻辑资源标签：`logical_t_count` / `logical_t_depth` / `logical_qubits`（每 ε 一行） |
| `processed_data/fault_tolerant_resources.parquet` | 物理资源标签：码距、物理比特、运行时间、总错误率（每 QEC 配置一行） |
| `processed_data/qec_model_config.json` | QEC 模型参数快照（可复现性） |
| `results/evaluation/` | 指标表（`metrics_raw_v8*.csv` 三套配置）、预测明细、统计检验、标签自检结果 |
| `results/figures/` | 论文图（PNG，7 张） |
| `scripts/` | 全流程脚本 |
| `docs/` | 数据集卡片、阶段报告、交叉核验方法学、远程迁移清单 |

## 3. 标签口径（论文必须按此表述）

普通线路的 `gate_count` **不是** FTQC 资源。本数据集把三层分开：

1. **算法 / 逻辑层**（`logical_resources.parquet`）
   - `logical_t_count(ε) = t_count_clifford_part + Σ_i T_cost(θ_i, ε)`
   - `t_count_clifford_part`：θ = kπ/4 的旋转所对应的**精确** T 门数（k 为奇数 → 1 T，k 为偶数即 Clifford → 0 T）
   - 非 Clifford 角用 Ross–Selinger / gridsynth 逐角合成，`T_cost(θ, ε)` 由 `qiskit.synthesis.discrete_basis.ross_selinger.gridsynth_rz` 实测
   - `t_depth_structural`：每个旋转只占 1 个 T 层（与 ε 无关的结构量）
   - `t_depth_serial`：每个量子比特上所有旋转 T 代价连续排列的串行下界
   - `logical_qubits = num_qubits`
2. **容错实现层**：魔法态工厂数 `ceil(t_count / rounds)`，15-to-1 蒸馏（`FACTORY_PATCHES=15`）
3. **物理架构层**（`fault_tolerant_resources.parquet`）
   - surface code 阈值标度 `p_L = 0.03 · (p/1e-2)^((d+1)/2)`，取满足总错误预算的最小奇数码距
   - `physical_qubits = 2d² · (logical_qubits + factories · 15)`
   - `runtime_seconds = rounds · d · 1µs`

⚠️ `fault_tolerant_resources.parquet` 是**解析参考估计器**（analytic reference estimator），
不是 Microsoft Quantum Resource Estimator 的读数。论文中必须这样标注；后续阶段会用 QRE / pyLIQTR / Qualtran 做交叉核验。

⚠️ **`c_if` 经典反馈线路不纳入数据集**：这类线路带中间测量，T-count 依赖运行时测量结果、不构成单一数值。
本轮实测 74 条，逐条报 `KeyError: <Clbit register=..., index=0>`（qiskit 无法展开到酉基），属"超出可定义范围"而非性能问题。

## 4. 可复现性

- 环境版本：见 `requirements.txt` 与 `processed_data/qec_model_config.json`
- 每条线路保留 `qasm_sha256`，可用于去重与溯源
- 角度合成缓存统计：`processed_data/synthesis_calibration.json`（unique_angles / cache hits）
- 随机种子：`42, 123, 2026, 2027, 2028`
- 所有失败/跳过组合留档：`logs/acquisition_failures*.csv`、`logs/feature_errors.csv`、`logs/label_timeouts.csv`
- 数据写入前自动备份：`circuits_manifest.backup_before_randomcircuit.parquet`、`*.backup_before_qpy.parquet`

## 5. 当前实验结果（阶段一，详见 `docs/STAGE1_REPORT.md`）

留一算法（LOSO，训练时完全不含该算法，32 个算法）：

| target | ridge | rf | gbdt |
|---|---|---|---|
| logical_t_count（R² / MdAPE%） | −2.421 / 79.40 | 0.998 / **0.79** | 0.996 / 0.86 |
| physical_qubits | 0.335 / 27.08 | 0.928 / 9.58 | 0.952 / **8.22** |

规模外推（训练 ≤16 比特 → 测试 >16 比特）：`logical_t_count` 上 rf 0.996（MdAPE 2.68%）、gbdt 0.996（3.10%），
ridge 为 −2.705（83.9%，38% 样本偏离超 10 倍）。

跨源外推（MQT Bench **1,246** 条训练 → QCircuitBench 6,236 条测试，后者从未进入训练）：

| target | ridge | rf | gbdt |
|---|---|---|---|
| logical_t_count（R² / MdAPE% / 行级偏离>10x） | −0.571 / 282.3 / **0.963** | 0.923 / 28.3 / 0.291 | 0.941 / **24.2** / 0.195 |
| physical_qubits | −25.22 / 387.9 / 0.622 | 0.780 / **23.4** / 0.000 | 0.743 / 29.5 / 0.000 |

结论：树模型在留一算法、规模外推与跨源外推上都可迁移，线性模型在 log 空间无界外推会崩溃；
`logical_qubits` 在设计上是 `num_qubits` 的恒等量，训练时已屏蔽 `num_qubits`，其规模/跨源外推 R² 为负属预期。

⚠️ **跨源 T-count 对训练集规模高度敏感，且"剔除样本"这条路已被实验否证**。三档训练规模（702 / 881 / 1,246 条）
给出完全不同的跨源结论（11.3% / 25.0% / 28.3%，最优模型也从 RF 翻到 GBDT）。本轮做了两个定向剔除消融：

| 配置（7,502 基座） | 训练侧处理 | RF | GBDT |
|---|---|---|---|
| v8 主表 | 无（1,246 条） | 28.31 | 24.19 |
| v8nc | 剔 1,068 条全 Clifford | 26.59 | 26.56 |
| v8ndyn | 剔动态四族 | 25.34 | 16.77 |

**没有一种剔除能回到 11.3%，且同一操作在 7,482 基座上方向相反**（详见报告 §4.9）。因此主表保留完整配置，
三套并列作为敏感性分析——**论文不能只报对自己有利的那一套，否则会被判选择性报告**。

统计显著性（`scripts/stats_tests.py`，簇 = `circuit_id`，B=2000，Holm 校正，12/12 显著）：

| 协议 | target | 配对 | n | 中位差(log10) | p(Holm) | 更好者 |
|---|---|---|---|---|---|---|
| grouped | T-count | rf−ridge | 1683 | −0.2923 | <1e-4 | rf |
| grouped | T-count | xgb−rf | 1683 | +0.0008 | <1e-4 | rf |
| grouped | physical | xgb−rf | 1900 | +0.0109 | <1e-4 | rf |
| crosssrc | T-count | xgb−rf | 5299 | +0.1287 | <1e-4 | **rf** |
| crosssrc | physical | xgb−rf | 6236 | −0.0585 | <1e-4 | xgb |

Friedman 三模型整体检验（块 = 线路）：四个 (协议, target) 组合均 p < 1e-4。平均秩最优为
grouped T-count rf 1.20 / grouped physical rf 1.21 / 跨源 T-count **rf 1.30（xgb 1.72）** / 跨源 physical rf 1.55（xgb 1.71，实质并列）。

⚠️ **跨源 T-count 的口径分裂**：GBDT 行级 MdAPE 更低（24.2% vs 28.3%），但配对中位差与 Friedman 秩偏向 RF；
>10 倍尾部 GBDT 明显更好（4.96% vs 13.9%），>2 倍比例两者几乎相同（33.5% vs 33.7%）。
**应表述为"GBDT 尾部更好、RF 中位更稳"**，不能单说某个模型跨源全面占优。

同源的 RF/GBDT 差异虽显著但量级极小（0.5%），论文里应表述为"同一量级、RF 略稳"。

种子方差（`scripts/seed_variance.py`）：**只有 grouped 有 5 个种子**，loso/scale/crosssrc 是确定性划分。
grouped 上树模型的 MdAPE 种子相对标准差 2–21%（物理比特 ±0.14 / ±0.05 pct、runtime ±0.05 / ±0.36 pct）；
T-count 的 RF 相对标准差达 48.8%，但它绝对值只有 0.0094%（绝对波动 ±0.0046 pct）——分母已落到仪器精度量级，
**报 grouped 指标必须同时给绝对值与标准差**。误差比例与 95% CI 见图 7。

## 6. 已知限制

- 训练集仍是**单源**：MQT Bench **1,246** 条。跨源测试集 6,236 条（QCircuitBench，协议 D）已不是瓶颈，
  训练侧的**多样性**才是；QASMBench 的 codeload tarball 在本网络三次截断，QDataSet 14 TB 未下载
- 动态控制流线路（`dynamic_qft`/`ghz_dynamic`/`ae`/`iqpe`）无法导出 OpenQASM2，已改用 **QPY 精确序列化**补采
  （`scripts/acquire_dynamic.py`，168/168 比特数校验一致），加载统一走 `scripts/circuit_io.py`
- **74 条"中间测量 + `c_if` 经典反馈"线路概念上不可标注**（T-count 依赖运行时测量、无单一定义），
  已从数据集排除并在 `logs/label_timeouts.csv` 留档；**不要再用"放宽 timeout"重试**
- MQT Bench 的 ALG 层 n>20 仍在 `ALG_LABEL_MAX_SIZE=20` 处截断（61 条），另有 17 条大 n INDEP 线路超时或永久卡死（`grover n≥30`）
- QEC 模型是解析近似，只用于"AI 能否学习配置到资源的映射"这一研究问题，不作为物理结论
- **扩样不保证更好**：702 → 881 → 1,246 条让跨源 T-count 在 11.3%–28.3% 区间漂移；两个定向剔除消融均不能复现 11.3%
  （详见报告 §4.9）。**"清洗子集"不是解法，唯一有效方向是提高训练侧的规模带与多样性**
- QCircuitBench 已导入 6,236 条（≤60 比特，占库内 16,589 条的 37.6%），其余可继续按需导入
- 统计检验已完成 Cluster Bootstrap + 配对 Wilcoxon + Friedman + Holm + 种子方差（仅 grouped 有 5 seeds）+ 图 7（误差比例 CI）；
  多任务 / 消融 E1–E5 未开始；GNN 第一版已跑通（`scripts/train_gnn.py`，见报告 §4.7）但单 seed、702 条规模且过拟合，待按 1,246 条重跑
- **独立合成器交叉核验（实现级）已完成**：1,500 组 (θ,ε) 角度级对账，10% 内 ≥99% 一致、ε 合规 0 越界（`docs/XCHECK_INDEPENDENT_SYNTHESIS.md`，报告 §4.8）；**算法级独立仍缺**（两端同属 Ross–Selinger，Qualtran/MS QRE 未接）
- 远程算力实测：新实例（128 核 / 503 GB / RTX 4080 SUPER）可用，但 **GPU 全程闲置**——训练是 sklearn 的 RF/GBDT/Ridge（8 核 5 seeds <10 min），
  真正需要 Linux 的只有 pyLIQTR；阶段二应选 8–16 vCPU 无 GPU，GPU 留到阶段三。详见 `docs/REMOTE_MIGRATION.md` §5b
