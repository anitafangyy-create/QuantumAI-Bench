#!/usr/bin/env python3
"""合成精度的物理不变式检查：T-count 应随 log2(1/ε) 线性增长。

为什么单独查这一条：
  Ross–Selinger 逐角合成的 T 成本满足 T_i ≈ 3·log2(1/ε) + O(1)（每角一条 3log2 的斜率）。
  若我们的 ε 标签真按同一口径合成，那么同一线路在 ε=1e-3/1e-6/1e-9 三档上的
  ΔT-count / Δlog2(1/ε) 应接近"非 Clifford 角数量 × 3"，而不是 1 或 5。
  这是不依赖任何外部工具、只依赖标签自身结构的物理一致性检查；斜率明显偏离，
  说明角度分类或 T 成本映射有错。

产出：results/evaluation/eps_scaling_check.json + 控制台摘要
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
EVAL = ROOT / "results" / "evaluation"

EPS = [1e-3, 1e-6, 1e-9]


def main() -> None:
    lab = pd.read_parquet(PROC / "logical_resources.parquet")
    feat = pd.read_parquet(PROC / "circuit_features.parquet")[
        ["circuit_id", "num_qubits", "n_nonclifford_rz", "t_count_clifford_part", "source", "algorithm"]
    ]
    wide = lab.pivot_table(index="circuit_id", columns="synthesis_epsilon", values="logical_t_count")
    missing = [e for e in EPS if e not in wide.columns]
    if missing:
        raise SystemExit(f"缺 ε 档：{missing}")
    df = wide.reset_index().merge(feat, on="circuit_id", how="left")

    # 只保留三档都有、且有非 Clifford 角的线路（全 Clifford 线路 ΔT 恒为 0，不提供斜率信息）
    df = df[(df.n_nonclifford_rz > 0) & df[EPS].notna().all(axis=1)].copy()
    for e in EPS:
        df[e] = pd.to_numeric(df[e], errors="coerce")
    df = df.dropna(subset=EPS)

    d63 = (df[1e-6] - df[1e-3]) / np.log2(1e-3 / 1e-6)      # ΔT / Δlog2(1/ε)，ε 收紧时 T 增大，斜率应为正
    d96 = (df[1e-9] - df[1e-6]) / np.log2(1e-6 / 1e-9)
    per_angle_63 = d63 / df.n_nonclifford_rz
    per_angle_96 = d96 / df.n_nonclifford_rz

    summary = {
        "n_circuits": int(len(df)),
        "per_angle_slope_1e-3_to_1e-6": {
            "median": float(np.median(per_angle_63)),
            "p10": float(np.percentile(per_angle_63, 10)),
            "p90": float(np.percentile(per_angle_63, 90)),
        },
        "per_angle_slope_1e-6_to_1e-9": {
            "median": float(np.median(per_angle_96)),
            "p10": float(np.percentile(per_angle_96, 10)),
            "p90": float(np.percentile(per_angle_96, 90)),
        },
        "expected_theory_slope_per_angle": 3.0,
        "frac_within_2p5_to_3p5": {
            "1e-3_to_1e-6": float(np.mean((per_angle_63 > 2.5) & (per_angle_63 < 3.5))),
            "1e-6_to_1e-9": float(np.mean((per_angle_96 > 2.5) & (per_angle_96 < 3.5))),
        },
        "note": "斜率按'每个非 Clifford 角每 log2(1/ε) 一位'归一；理论值 3（Ross–Selinger 的 3log2 界）。",
    }
    # 分算法看一眼异常族
    by_algo = (df.assign(slope=per_angle_96)
                 .groupby("algorithm").agg(n=("slope", "size"), median_slope=("slope", "median"))
                 .sort_values("n", ascending=False).head(15))
    summary["worst_algos"] = by_algo[by_algo.median_slope.abs() > 0.5].to_dict(orient="index")

    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "eps_scaling_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
