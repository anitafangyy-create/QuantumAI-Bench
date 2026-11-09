# 迁移到远程 Linux 主机：执行清单与实测结论

> 本文件记录两件事：① **本项目实际做过的一次远程迁移尝试及其失败原因**（2026-11-04 夜，AutoDL 风格实例）；
> ② 迁移到远程机器时**必须照做的清单与验收闸门**（数字已同步到当前数据集 6,938 线路这一版）。

## 0. 实测结论（先看这段）

| 项 | 事实 |
|---|---|
| 远端规格 | 128 核 / 503 GB RAM / RTX 4080 SUPER 32 GB；`/root` 30 GB（剩 15 GB）、`/root/autodl-tmp` 186 GB（剩 23 GB）；Python 3.10，清华 pypi 镜像通，`rsync` 可用 |
| 连通的验证 | SSH 端口转发可用；项目 869 MB 用 **tar 流式**（`tar -czf - ... \| ssh host "tar -xzf - -C ..."`）成功落地；`qenv`（py3.12）与 `liqtr-env`（py3.11 + pyLIQTR）均装通；`torch 2.6.0+cu124`，`torch.cuda.is_available() == True` |
| 失败点 | 编排脚本在 `build_dataset.py --extend` 生成 **grover n=36** 时卡死：18 分钟 97.9% CPU、无任何输出、无新 qasm 落盘 |
| 后果 | `kill -9` 后 SSH 断开，随后 `Connection refused` 持续约 20 分钟未恢复（实例疑似被关机/回收）。**远端环境与日志全部失联**，只剩一次 `scp` 测试文件 |
| 结论 | **当前不是"要不要上 GPU"的问题，而是"换机器也会卡在同一处"**：根因是 qiskit Rust transpiler 在 n≥30 的 grover 上不响应 `SIGALRM`，任何机器都会挂在这一个点上 |

**因此本阶段的正确顺序是：先把会卡死的采集步骤从关键路径上摘掉，再谈迁移。**

- 采集/标注/特征/QEC 全是**单线程或弱并行 CPU 任务**，租 GPU 跑它们等于烧钱（GPU 全程闲置）；
- 真正吃 GPU 的只有阶段三的 GNN / 多任务 / 消融，而 `train_gnn.py` 目前连本机都还没跑通；
- 本机（8 核 / 8 GB）在**已标注 6,938 条**的规模上，四协议训练 + 统计 + 出图约 40 分钟一轮，仍属可接受。

## 1. 迁移原则

| 内容 | 本机 | 远程 |
|---|---|---|
| 线路采集、T-count 标注、QEC 标签 | ✅ 留在本机（已验证 6,938 条可跑完） | ❌ 不要重跑（纯 CPU，租 GPU 是浪费） |
| 特征工程、EDA、画图 | ✅ | 可选 |
| 表格基线 + GNN/多任务训练、超参搜索 | 小规模可留 | ✅ 主体（**这正是需要 GPU 的部分**） |
| 大规模导入（QCircuitBench 剩余部分） | ✅ 已用 `--workers 6` 跑通（6,000 条 / 34 分钟） | ✅ 128 核可把 `--workers` 提到 24–32 |

**要搬的只有四样东西**（总量 < 1 GB）：

```
scripts/                     # 全部脚本
processed_data/*.parquet     # 三层标签表（features / logical / fault_tolerant）
raw_data/circuits_manifest.parquet
raw_data/circuits/*.qasm     # 7,392 个，约 800 MB
requirements.txt
```

**不要**搬 `raw_data/QDataSet/`（14 TB 未下载）与 `raw_data/QASMBench/`（不完整）。

## 2. 同步命令（本机实测可用）

本机 `rsync` 是 **openrsync 2.6.9**，不认 `--info=progress2` 等新参数（会报 usage）；`scp` 需要密码交互。
实测最稳的是 **tar 流式**：

```bash
cd /Users/fangyuanyuan/Downloads/机构项目/量子计算机/QuantumAI-Bench
tar -czf - scripts processed_data raw_data/circuits_manifest.parquet raw_data/circuits requirements.txt \
  | ssh -p <port> root@<host> "mkdir -p /root/autodl-tmp/qbench/QuantumAI-Bench && tar -xzf - -C /root/autodl-tmp/qbench/QuantumAI-Bench"
```

免密建议：本地生成专用密钥并把公钥写入远端 `authorized_keys`，再在 `~/.ssh/config` 里加别名
（`IdentityFile` 指向该密钥、`StrictHostKeyChecking no`、`ServerAliveInterval 30`），避免每次输密码。

> `build_dataset.py` 用 `Path(__file__).resolve().parents[1]` 取根目录，**保持 `scripts/` 与 `raw_data/`、`processed_data/` 同级**即可，无需改代码。

## 3. 远程环境（固定版本，避免与本地漂移）

```bash
conda create -p /root/autodl-tmp/qenv python=3.12 -y
/root/autodl-tmp/qenv/bin/pip install -r /root/autodl-tmp/qbench/QuantumAI-Bench/requirements.txt
```

Linux 上 xgboost 可以正常加载（不需要 libomp），因此 `train_models.py` 会走 `XGB_BACKEND="xgboost"`，
与本地 `sklearn_hist_gbdt` 的数值**不能直接对比**，报告中必须分别标注后端。

阶段三额外装（实测 `cu124` 在 4080 SUPER 上可用）：

```bash
/root/autodl-tmp/qenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 4. 迁移后的验收闸门（必须全过才算搬成功）

```bash
cd /root/autodl-tmp/qbench/QuantumAI-Bench

# ① 数据完整性：数字必须与本地完全一致
.venv/bin/python scripts/report_stats.py

# ② 标签自检：4 项全 passed
.venv/bin/python scripts/verify_labels.py

# ③ 物理不变式：斜率中位应仍在 3.0 附近
.venv/bin/python scripts/check_eps_scaling.py

# ④ 训练冒烟：单 seed 单模型，确认数据能合并、无 NaN
.venv/bin/python scripts/train_models.py --models xgb --protocols grouped --seeds 42

# ⑤ 出图：6 张图不报错
.venv/bin/python scripts/make_figures.py
```

第 ① 步的对照基准（本机 2026-11-05 实测，**与远端必须逐项相等**）：

| 指标 | 期望值 |
|---|---|
| manifest 行数 | 7274 |
| 已标注线路 | 6938（ALG 211 / INDEP 491 / EXT 6236） |
| 逻辑标签行数 | 20,814 |
| 物理标签行数 | 187,326 |
| 算法数 | 190 |
| 配置数 | 27 |

任何一项对不上，先查 `raw_data/circuits_manifest.parquet` 与 `processed_data/` 是否完整传完，**不要**直接重跑标注。

## 5. 远端编排脚本与已知雷区

| 脚本 | 用途 | 状态 |
|---|---|---|
| `scripts/night_run.sh` | v1 全量（含 MQT 扩网格） | ❌ **已知会在 `grover n=36` 永久卡死**，不要再用 |
| `scripts/night_run2.sh` | v2（去掉 MQT 扩网格，只做导入→标注→QEC→训练→统计→图） | ✅ 推荐；远端恢复后直接跑这个 |
| `scripts/local_run.sh` | 本机编排 | ⚠️ 其 label 阶段会反复重试 336 条 ALG `n>12` 截断项（设计上就标不出来），跑到该阶段需手动停 |

**核心雷区**：`build_dataset.py --size-grid ... 36 40` 生成 `grover n≥30` 时，qiskit Rust transpiler 不响应
`SIGALRM`，任何 `--timeout` 都不生效，进程永久 97.9% CPU。要在远端扩 MQT 样本，必须先按算法白名单排除
`grover` 的大 n 档，或改成"先生成小 n、再离线单独处理"。


## 5b. 2026-11-06 第二次远端实测（新实例 39776，专门为 pyLIQTR 交叉核验而开）

| 项 | 事实 |
|---|---|
| 规格 | 128 核 / 503 GB / RTX 4080 SUPER 32 GB；系统盘 30 GB（**已用 22 GB，只剩 8.4 GB**，注意先清 qasm 再扩容） |
| 环境 | `conda create -n liqtr python=3.10` + `pip install pyLIQTR pygridsynth` **一次通过**；`conda create -n qbench python=3.10` + `qiskit 2.6.2 / mqt-bench` 通过 |
| 交叉核验 | 成功（见 `docs/XCHECK_INDEPENDENT_SYNTHESIS.md`）：1,500 组 (θ,ε)，中位 T 数一致、ε 合规 0 越界 |
| 传文件 | **不要用 paramiko**（本机 `cryptography` 需 Rust 编译，装不上）；用 `expect` 包 `sftp put`，141 MB 约 58 s（2.4 MB/s）。**expect 里密码分支不能用 `exp_continue` 后直接等提示符**，否则会一直等第二次密码提示——必须拆成两段 expect |
| 扩样本结果 | `build_dataset.py --extend --levels INDEP --random-reps 8` 生成耗时仅 66 s，但 **INDEP 已饱和**：36 个算法 × 25 个规模 = 612 个唯一格（每格 rep=1），加 rep 只多 46 行；最终仅 **+31 条可标注**（6969 线路 / 20907 标签行） |

**本轮新增雷区（重要）**：`build_dataset.py --extend` **不是"只生成"**——它会顺带重跑完整标注，
对没有 qasm 的 6,236 条 QCB 行全部判失败，并把 `circuit_features.parquet` / `logical_resources.parquet`
**覆盖成只剩 31 / 93 行**。远端扩容必须先在旁路解包原件备份，跑完再恢复，或改成
"只生成 → 单独跑 `label_parallel.py --source MQT_Bench`"。

**结论**：这台机器的价值不在"更快"，而在**能装 Linux-only 依赖**（pyLIQTR）与**能并行标注**。
就数据规模而言，MQT INDEP 已饱和、ALG 又被 `ALG_LABEL_MAX_SIZE=12` 截断，
**继续堆规模的唯一轴是放开 ALG 截断（n=13–20 还有 118 条线路可标注）**，这属于口径决策，需要用户拍板。

## 6. 分阶段机器选型

| 阶段 | 数据规模 | 建议配置 | 备注 |
|---|---|---|---|
| 阶段一（**已完成**） | 6,938 线路 / 187,326 记录 | 本机即可 | 导入 6,000 条 34 min（6 workers）+ 标注 + 四协议训练 40 min |
| 阶段二 | 训练集扩到 2k–3k 线路 / 500k+ 记录 | 8–16 vCPU、32 GB RAM、无 GPU | RF/GBDT 与特征工程是 CPU 主导；128 核主要加速**导入**这一步 |
| 阶段三 | 同上 + GNN/多任务/消融 | 同上 + 1 块 ≥16 GB 显存 GPU | 100k 图 × 5 seeds × 5 组消融 |
| 方向 D（可选） | LLM 生成线路 | 多卡或租用 | 只有这条线必须大显存 |

按小时计费的 GPU 机器**不要**用来做标注、特征工程或表格基线训练。

## 7. 已知会随环境变化的东西

| 项 | 本机（macOS） | 远程（Linux） |
|---|---|---|
| `xgboost` | 缺 libomp，降级 `HistGradientBoostingRegressor` | 正常 |
| `rsync` | openrsync 2.6.9，不支持 `--info=progress2`；建议 tar 流式 | GNU rsync，参数齐全 |
| 进程管理 | `pkill`/`ps aux` 受限；`kill -9` 会触发确认弹窗，长任务需用受管后台任务 | 正常 |
| 命令超时 | 无 `timeout` 命令，用子进程 `proc.kill()` | 可用 `timeout`（但对 grover 无效，见 §5） |
| 长任务稳定性 | 沙箱曾多次掉线中断训练 | 更适合无人值守 |
