# 夜间运行纪要（2026-11-04 夜 → 11-05 早）

> 一句话：**远程 GPU 主机迁移失败（宿主卡死后失联），全部工作已在本机完成——数据集扩到 6,938 条线路 / 187,326 条记录，
> 四协议重训 + 统计检验 + 6 张图 + 5 项标签自检全部跑通，四份文档已同步到这一版。**

## 1. 时间线

| 时间（本地） | 事件 |
|---|---|
| 11-04 晚 | 拿到远程 GPU 主机 SSH（128 核 / 503 GB / RTX 4080 SUPER 32 GB），完成连通性、免密、869 MB 项目传输、两套 conda 环境（`qenv` py3.12 + `liqtr-env` py3.11）、`torch.cuda.is_available()=True` |
| 11-04 23:40 前后 | 启动夜间编排；`build_dataset.py --extend` 在 **grover n=36** 上 18 分钟 97.9% CPU 无输出（qiskit Rust transpiler 不响应 `SIGALRM`） |
| 11-04 23:55 起 | `kill -9` 后 SSH 断开，`Connection refused` 持续约 20 分钟未恢复 → **实例失联，远端数据/环境全部不可用** |
| 11-05 00:10 → 01:40 | 改造 `import_qcircuitbench.py`：`--workers`（并发 chunk）+ `--skip-processed`（断点续跑）；本机导入 6,000 条 |
| 11-05 01:40 → 02:10 | 导入完成：**5,757 成功 / 243 失败（4.05%）**，用时 2,063 s（约 0.34 s/条，6 workers） |
| 11-05 02:10 → 03:30 | 标注：**6,938 条 / 20,814 行**；剩余 336 条全部是 ALG `n>12` 设计内截断，重试无意义 |
| 11-05 03:31 → 03:47 | 标签自检 4 项全 passed；新增 ε 斜率检查：1e-6→1e-9 每角斜率中位 **3.05**（理论 3） |
| 11-05 03:47 → 04:10 | QEC 重建：**187,326 行**，全部 feasible，码距 3–27 |
| 11-05 04:10 → 04:50 | 首次四协议训练在 crosssrc 报 `inf`（线性模型预测上溢），定位并修复 `metrics()` 的 `expm1` 溢出后重训成功 |
| 11-05 04:50 → 05:20 | 统计（Friedman + 配对 Wilcoxon + Cluster Bootstrap + Holm）、6 张图重绘、`dataset_stats.json` 重出 |
| 11-05 05:20 → 06:00 | 四份文档（`STAGE1_REPORT.md` / `DATASET_CARD.md` / `README.md` / `REMOTE_MIGRATION.md`）逐节同步；6 张图逐张核验 |

## 2. 当前数据规模（权威值，来自 `results/evaluation/dataset_stats.json`）

| 项 | 数值 |
|---|---|
| manifest | 7,274 行（MQT Bench 1,038 + QCircuitBench 6,236） |
| 已标注线路 | **6,938**（ALG 211 / INDEP 491 / EXT 6,236），已标注率 **95.4%** |
| 逻辑标签 | **20,814** 行（×3 档 ε） |
| 物理/QEC 标签 | **187,326** 行（×27 配置，全 feasible） |
| 算法 / 比特 | **190 个 / 2–60** |
| T-count 范围 | 0 – 12,806,100（全 Clifford 线路 1,047 条） |
| 码距 / 物理比特 / runtime | 3–27 / 36 – 4.31e5 / 6e-6 – 64.0 s |

**阶段一规模目标（10,000–100,000 样本）已基本达成**：线路数进入 5k–10k 区间上沿，记录数超 10 万目标 87%。

## 3. 关键结果（5 seeds 均值；`gbdt` 本机为 sklearn HistGBDT 后端）

分组内（协议 A）：T-count RF **0.999 / 0.375%**、xgb 0.999 / 0.475%、ridge 0.644 / 42.6%；
物理比特 RF 0.989 / 2.15%、xgb 0.992 / 3.01%、ridge 0.713 / 20.9%。

跨源（协议 D，训练 MQT 702 → 测试 QCircuitBench **6,236 条 / 168,372 行**）：

| target | ridge | rf | gbdt |
|---|---|---|---|
| logical_t_count（R² / MdAPE% / 行级 >10x） | −0.537 / 282.3 / 0.919 | **0.947 / 11.3 / 0.245** | 0.933 / 22.1 / 0.339 |
| physical_qubits | −25.43 / 387.9 / 0.621 | 0.702 / 30.2 / 0.001 | 0.650 / 29.6 / 0.002 |
| runtime_seconds | −1.31 / 152.1 / 0.620 | 0.588 / 32.9 / 0.001 | 0.431 / 68.0 / 0.170 |

统计（簇 = `circuit_id`，B = 2000，Holm）：四个 (协议, target) 组合 Friedman p 全部 < 1e-4，最优全部是 RF
（跨源 T-count 平均秩 rf 1.07 / xgb 1.97 / ridge 2.96；跨源物理 rf 1.69 / xgb 1.73 / ridge 2.59）。

**两条必须写进论文的口径警告**（已写入报告 §4.4）：
1. 跨源集 956 → 6,236 条后 MdAPE 由 20.9% 降到 11.3%，**是测试面组成变化，不是模型改进**；
2. **行级与线路级误差比例差异巨大**（跨源 T-count 行级 >10x 为 RF 24.5%，线路级为 0）——两个数必须同时报。

## 4. 本轮新增/修复

| 文件 | 变化 | 状态 |
|---|---|---|
| `scripts/import_qcircuitbench.py` | 新增 `--workers`（并发 chunk）、`--skip-processed`（断点续跑） | 6,000 条导入验证通过 |
| `scripts/train_models.py` | `metrics()` 内把 `y_pred_log` 钳到 `[-50, 20]`，`expm1` 后非有限值兜底 | 修复跨源 `inf` 崩溃 |
| `scripts/check_eps_scaling.py` | 新建：ε→T-count 每角斜率不变式（理论 3log2） | 3,911 条，渐近区中位 3.05 |
| `scripts/train_gnn.py` | 新建 GNN 基线（门级 DAG 消息传递）；修了 3 处：旧 qelib 门（`rxx`）兜底解析、节点特征维与模型输入维对齐、torch 环境独立 | **已跑通**（`.venv-gnn`，torch 2.15 CPU）：grouped R² 0.984 / MdAPE **21.9%** / >10x 2.9%（表模型 RF 0.375%），单 seed、明显过拟合，属负结果 |
| `docs/REMOTE_MIGRATION.md` | 重写：记录本次迁移实测失败 + 更新验收闸门数字 | 完成 |

## 5. 仍未解决（明天要先决的）

1. **独立估计器交叉核验**（pyLIQTR / Qualtran / MS QRE）——唯一还没补的外部可信度，pyLIQTR 只在已失联的远端装过。
2. **训练集只有 702 条（MQT）**——跨源测试集已很大，但训练侧是真正瓶颈；扩 MQT 必须先绕开 `grover n≥30` 的永久卡死。
3. **GNN 只是第一版基线**：已跑通但过拟合（train MSE 0.009 vs val 0.29–0.44），需扩训练集 + 5 seeds + 正则后再下 E1 结论。
4. **远端主机是否还能重启**、重启后端口/密码是否变化，未知。
5. **ALG `n>12` 截断口径**（接受截断 vs 只保留 INDEP 层）尚未与你确认。

## 6. 次日补充（11-05 上午）

| 事项 | 结果 |
|---|---|
| pyLIQTR 交叉核验（本机接入） | **失败**：`pip install pyLIQTR` 拉 pygridsynth → numba → llvmlite，macOS/py3.12 无可用 wheel，本地编译 LLVM 报 `CalledProcessError`。与早期结论一致：这条路在 macOS 上不可用，需 Linux 才能跑 |
| pytket 替代方案 | 已装 pytket 2.18.5 只到 `CX+TK1` 层，`AutoRebase` 到 Clifford+T 报 "No known decomposition from TK1"，无法给出独立 T-count |
| 种子方差 | **已做**（`scripts/seed_variance.py`）：只有 grouped 有 5 个种子，树模型 MdAPE 相对标准差 5–12%（T-count ±0.04 pct、物理比特 ±0.15 pct）；loso/scale/crosssrc 为确定性划分，不能声称种子稳定性 |
| 图 7 | **已出**（`scripts/make_figure7.py`）：>2 倍误差比例 + 95% cluster bootstrap CI 四宫格，跨源尾部结构一眼可见 |

## 7. 次日补充（11-06：数据集 6,938 → 7,117，报告正文按新数据改写）

| 事项 | 结果 |
|---|---|
| 336 条未标注的真实原因 | **不是 `ALG_LABEL_MAX_SIZE` 截断**，而是 `no_qasm_export: 动态控制流线路无法导出 OpenQASM2`（`dynamic_qft`/`ghz_dynamic`/`ae`/`iqpe` 四族，qiskit QASM2 导出器直接拒绝）。这些线路从来没有落盘文件 |
| 解法 | QASM3 只救回 2 族（`ghz_dynamic` 报 `binary operation '^'`、`ae` 报 `TypeError`）；改用 **QPY**（qiskit 原生序列化）四族全部往返一致（`scripts/acquire_dynamic.py`，168/168 成功、0 比特数不符） |
| 新代码 | `scripts/circuit_io.py`（QASM2/QASM3/QPY 统一加载）、`acquire_dynamic.py`；`build_dataset.py` / `label_worker.py` / `label_parallel.py` / `verify_labels.py` 全部改走 `load_circuit`；`ALG_LABEL_MAX_SIZE` 12→20 |
| 两个坑 | ① `label_worker.py` 漏 `import sys` → `NameError`，而 `label_parallel.py` 把 worker stderr 丢进 `DEVNULL`，表现为"日志一直刷 0 problems、一条没落地"（静默失败）；② `label_parallel.py` 比特预扫描按文本读文件头，遇 `.qpy` 抛 `UnicodeDecodeError` |
| 标注 | `circuits=7117 label_rows=21351 problems=152 elapsed=1956s`；MQT 层 702 → **881**（ALG 336 / INDEP 545），QCB 仍 6,236 |
| 结论翻转 | grouped 基本持平（T-count rf 0.24%/xgb 0.45%）；**跨源 T-count rf 11.3% → 25.0%（退化）、xgb 22.1% → 14.3%（改善）**，Friedman 平均秩最优由 rf 翻为 **xgb**（1.17 vs 1.83） |
| 翻转机制 | 新增样本里 21 条是"大 n 且 T≡0"的全 Clifford 线路（`bv` n≤28、`dj` n=14），训练侧 T≡0 样本升到 131 条（14.9%），强化"多比特 ⇒ 零 T"的**错误**先验；而 QCB 测试侧 T 中位 75,047 |
| 文档 | `docs/STAGE1_REPORT.md` 正文按 7,117 重写（§1/§2/§4.1–4.6/§5/§6/§7），新增 **§4.9 扩样敏感性分析**；`README.md`、`docs/DATASET_CARD.md` 同步 |

## 8. 11-06 夜间续跑：7,117 → 7,502 + 定向剔除消融（否证）

**用户三个决定已全部执行**：① 剔除（全 Clifford 训练样本）② 坚持（继续攻坚未标注）③ 补充（补 randomcircuit 同族新实例）。

**数据（最终）**：manifest 7,654 / 已标注 **7,502**（MQT 1,246 = ALG 459 + INDEP 787；QCB 6,236）
/ 逻辑 **22,506** / QEC **202,554**（码距 3–29、物理比特 36–4.31e5、runtime 6e-6–378 s）。三层完全对齐。
未标注 152 条：61 ALG `n>20` 设计跳过 + **74 条 `c_if` 概念不可标注** + 17 条大 n 超时。

**补充（③）**：`scripts/acquire_randomcircuit.py` 按 rep=1..20 生成同族新实例 402 条（原 39 条全是 rep=0），
分三批标注成功 397 条（含 n=24 的 20 条）。`randomcircuit` 在 MQT 侧从 39 → 397 条，留一误差未上升（2.6%），
说明族内多实例是安全增益。

**坚持（②）**：先标 360 条新 randomcircuit（`circuits=7477 problems=0 elapsed=1417s`），
再攻坚 79 条长期卡住的小规模（`--timeout 1200 --max-qubits 60`）→ 仅 +5 条，74 条报
`KeyError: <Clbit register=(3,"meas"), index=0>`。逐条确认是**中间测量 + `c_if` 经典反馈**，qiskit 无法展开到酉基，
其 T-count 依赖运行时测量结果、无单一定义 → **概念上不可标注，不再重试**。

**剔除（①）+ 两个消融（跨源 T-count，测试集恒为 6,236 条 QCB）**：

| 基座 | 配置 | 训练侧处理 | RF | GBDT |
|---|---|---|---|---|
| 7,502 | v8（主表） | 无 | 28.31 | 24.19 |
| 7,502 | v8nc | 剔 1,068 条全 Clifford | 26.59 | 26.56 |
| 7,502 | v8ndyn | 剔 `ae`/`dynamic_qft`/`ghz_dynamic`/`iqpe` | 25.34 | **16.77** |
| 7,482 | v7 | 无 | 27.42 | 16.39 |
| 7,482 | v7nc | 剔全 Clifford | 28.60 | 19.97 |
| 7,482 | v7ndyn | 剔动态四族 | 27.92 | 18.43 |

**结论是否证**：没有一种定向剔除能回到 702 条时代的 11.3%，且同一操作在 7,482 / 7,502 两个基座上**效应符号相反**。
真主导变量是**训练集规模带整体下移**（702→1,246），不是任一子集。因此主表保留完整 1,246 条，三套配置并列作敏感性分析。
**"全 Clifford 样本毒化先验"这个机制解释已被本轮实验推翻**，报告中相应段落已改写（§4.9、§5.4、§6、§7）。

**统计与图已按 7,502 重算**：`stats_tests.csv`（crosssrc T 5,299 块，12/12 Holm 显著）、`stats_friedman.csv`、
`stats_bootstrap.csv`、`seed_variance.csv`、`eps_scaling_check.json`（4,441 条，渐近区斜率 3.025）、fig1–fig7（23:44–23:45 落盘）。

**新增脚本/参数**：`acquire_randomcircuit.py`（新）；`label_parallel.py --ids-file`；
`train_models.py --drop-clifford-train / --drop-algos-train / --tag`；`verify_labels.py` 走 `load_circuit`；
`make_figures.py` fig5 脚注、`make_figure7.py` 脚注改为按数据实时计算。

**仍待拍板**：论文主表用 702 / 881 / 1,246 哪一档；GNN 需按 1,246 条重跑；grouped 的 T-count 相对标准差是否改报绝对波动。
