#!/usr/bin/env python3
"""标签独立核验：用与主流水线不同的路径重算 T-count / T-depth，检查一致性。

核验项：
  1. Clifford-only 线路（GHZ）的 T-count 必须为 0
  2. 用 qiskit transpile 到 {h,t,tdg,s,sdg,x,z,cx} 基（其内部 gridsynth 默认 ε=1e-10）
     的 T 门数，与流水线在 ε=1e-10 下逐角合成的结果对比
  3. T-count 的 ε 单调性：ε 越小 T-count 不应减少
  4. 结构量 t_depth_structural ≤ 串行 T 深度（对 ε=1e-3）
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import transpile, qasm2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_io import load_circuit  # noqa: E402
from qiskit.synthesis.discrete_basis.ross_selinger import gridsynth_rz

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
EVAL = ROOT / "results" / "evaluation"


def t_count_via_qiskit_basis(qc) -> int:
    """独立路径：直接用 qiskit 的 Clifford+T 基展开并数 T 门。

    必须同时计 t 和 tdg：qiskit 的 Clifford+T 展开会把负角度实现成 tdg，
    只数 `t` 会系统性低估（实验中最多低估到 8/14）。
    """
    out = transpile(qc, basis_gates=["h", "t", "tdg", "s", "sdg", "x", "z", "cx"],
                    optimization_level=0, seed_transpiler=0)
    ops = out.count_ops()
    return int(ops.get("t", 0)) + int(ops.get("tdg", 0))


def t_count_via_gridsynth(qc, eps: float) -> int:
    """与流水线相同的分解路径，但在这里独立重算一遍。"""
    base = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0, seed_transpiler=0)
    total = 0
    for inst in base.data:
        if inst.operation.name != "rz":
            continue
        theta = float(inst.operation.params[0])
        k = round(theta / (math.pi / 4))
        if abs(theta - k * math.pi / 4) < 1e-8:
            km = k % 8
            total += 1 if km % 2 == 1 else 0   # 偶数 k 是 Clifford（I/S/Z/S†），0 个 T
        else:
            total += int(gridsynth_rz(theta, epsilon=eps).count_ops().get("t", 0))
    return total


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(ROOT / "raw_data" / "circuits_manifest.parquet")
    feats = pd.read_parquet(PROC / "circuit_features.parquet")
    labels = pd.read_parquet(PROC / "logical_resources.parquet")

    # 抽样：不同算法、不同规模
    sample = (feats[feats.abstraction_level == "INDEP"]
              .groupby("algorithm", group_keys=False)
              .apply(lambda g: g.nsmallest(1, "num_qubits"), include_groups=False))
    sample_ids = sample["circuit_id"].tolist()
    sample_ids = sample_ids[:8]

    rows, checks = [], []
    for cid in sample_ids:
        rec = manifest[manifest.circuit_id == cid].iloc[0]
        if not rec.qasm_path:
            continue
        qc = load_circuit(rec.qasm_path)
        t_qiskit = t_count_via_qiskit_basis(qc)
        t_grid_10 = t_count_via_gridsynth(qc, 1e-10)
        row = labels[(labels.circuit_id == cid)]
        t_pipeline = row[row.synthesis_epsilon == row.synthesis_epsilon.min()].iloc[0]
        rows.append({
            "circuit_id": cid, "algorithm": rec.algorithm, "num_qubits": int(rec.num_qubits),
            "t_count_pipeline_eps1e-3": int(t_pipeline.logical_t_count),
            "t_count_gridsynth_eps1e-10": int(t_grid_10),
            "t_count_qiskit_basis": int(t_qiskit),
            "ratio_gridsynth_vs_qiskit": (t_grid_10 / t_qiskit) if t_qiskit else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(EVAL / "label_cross_check.csv", index=False)

    checks.append({
        "check": "ghz_t_count_zero",
        "passed": bool((docs := feats[(feats.algorithm == "ghz") & (feats.abstraction_level == "INDEP")]).shape[0] > 0
                       and docs["t_count_clifford_part"].max() == 0 and docs["n_nonclifford_rz"].max() == 0),
        "detail": "GHZ 线路应完全由 Clifford 门构成",
    })

    # ε 单调性
    mono_violations = 0
    total_series = 0
    for cid, g in labels.groupby("circuit_id"):
        g = g.sort_values("synthesis_epsilon")
        if len(g) >= 2:
            total_series += 1
            if not np.all(np.diff(g.logical_t_count.to_numpy()) <= 0):  # eps 变小 → T 应不变或增大
                mono_violations += 1
    checks.append({"check": "t_count_monotonic_in_epsilon", "passed": mono_violations == 0,
                   "detail": f"violations={mono_violations}/{total_series}"})

    # 结构深度 <= 串行深度
    bad = int((labels.logical_t_depth < labels.t_depth_structural).sum())
    checks.append({"check": "t_depth_serial_ge_structural", "passed": bad == 0,
                   "detail": f"violations={bad}"})

    # 与 qiskit 基路径的一致性（ε=1e-10）。注意：两条路径的绝对 T 数不是唯一定义
    # ——qiskit 内建的 rz→Clifford+T 合成并非最优 gridsynth，因此允许 ≤20% 差异，
    # 并记录方向（qiskit 读数应 ≥ 逐角 gridsynth 读数）。
    if len(df):
        rel = (df["t_count_gridsynth_eps1e-10"] - df["t_count_qiskit_basis"]).abs() / df["t_count_qiskit_basis"].replace(0, np.nan)
        over = (df["t_count_qiskit_basis"] < df["t_count_gridsynth_eps1e-10"]).sum()
        checks.append({"check": "t_count_matches_qiskit_basis_at_eps1e-10",
                       "passed": bool(rel.max() < 0.20 and over == 0),
                       "detail": f"max_rel_diff={rel.max():.4f}, median={rel.median():.4f}, "
                                 f"qiskit_lower_than_gridsynth={int(over)}/{int(rel.notna().sum())}"})

    with open(EVAL / "label_checks.json", "w") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)
    print(df.to_string(index=False) if len(df) else "no sample")
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
