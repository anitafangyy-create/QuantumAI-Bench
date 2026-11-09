#!/usr/bin/env python3
"""阶段二：容错（QEC）+ 物理资源标签生成。

把「逻辑资源」（T-count / T-depth / Clifford / logical qubits）与
「QEC 配置」（物理错误率 p、逻辑错误预算、码距 d）和
「硬件配置」（码周期时间）组合，生成容错资源记录。

所有公式集中在此处，明确标注为「解析基准模型（analytic reference estimator）」，
用于给 AI 模型提供确定性标签，而不是声称等同于 Microsoft QRE / pyLIQTR 的读数。

模型假设（surface code, 标准阈值标度）：
  每轮逻辑错误率  p_L(d, p) = A * (p / p_th)^((d+1)/2),  A=0.03, p_th=1e-2
  逻辑操作层数      rounds_total = t_depth + clifford_depth
  总逻辑错误       p_total ≈ 1 - (1 - p_L)^{rounds_total}
  码距 d           取满足 p_total ≤ budget 的最小奇数 d（3..d_max）
  数据量子比特     2 d^2 每逻辑比特（surface code patch：data + measure）
  魔法态工厂       15-to-1 蒸馏，15 个 patch/工厂；工厂数按 T 吞吐需求估计
  运行时间         rounds_total * d * cycle_time（每逻辑层 d 个码周期）
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"

P_TH = 1e-2          # surface code 阈值
PREFACTOR = 0.03     # 阈值以上标度前置因子
CYCLE_TIME = 1e-6    # 超导码周期 1 us
FACTORY_PATCHES = 15 # 15-to-1 蒸馏工厂占用的 patch 数
D_MAX = 41


def logical_error_per_round(d: int, p: float) -> float:
    return PREFACTOR * (p / P_TH) ** ((d + 1) / 2)


def choose_distance(p: float, budget: float, rounds: int) -> tuple[int | None, float]:
    """选择满足总逻辑错误预算的最小奇数码距。"""
    best_d, best_err = None, 1.0
    for d in range(3, D_MAX + 1, 2):
        p_l = logical_error_per_round(d, p)
        p_total = 1.0 - (1.0 - p_l) ** max(rounds, 1)
        if p_total <= budget:
            return d, p_total
    return best_d, best_err


def build(dataset: pd.DataFrame, features: pd.DataFrame, params: dict) -> pd.DataFrame:
    feat = features.set_index("circuit_id")
    rows = []
    for rec in dataset.itertuples():
        if rec.circuit_id not in feat.index:
            continue
        f = feat.loc[rec.circuit_id]
        logical_qubits = int(f["num_qubits"])
        clifford_layers = int(f["circuit_depth"])
        rounds = int(rec.logical_t_depth) + clifford_layers
        for p in params["physical_error_rate"]:
            for budget in params["logical_error_budget"]:
                d, p_total = choose_distance(p, budget, rounds)
                status = "feasible" if d is not None else "infeasible_budget"
                if d is None:
                    rows.append({
                        "sample_id": f"{rec.circuit_id}|eps{rec.synthesis_epsilon:g}|p{p:g}|e{budget:g}",
                        "circuit_id": rec.circuit_id, "algorithm": rec.algorithm, "source": rec.source,
                        "abstraction_level": rec.abstraction_level, "num_qubits": logical_qubits,
                        "synthesis_epsilon": rec.synthesis_epsilon,
                        "physical_error_rate": p, "logical_error_budget": budget,
                        "code_distance": np.nan, "rounds_total": rounds,
                        "logical_error_total": np.nan, "physical_qubits": np.nan,
                        "runtime_seconds": np.nan, "factory_count": np.nan,
                        "status": status,
                    })
                    continue
                total_cycles = rounds * d
                n_factories = int(math.ceil(rec.logical_t_count / max(rounds, 1))) if rec.logical_t_count > 0 else 0
                physical = 2 * d * d * (logical_qubits + n_factories * FACTORY_PATCHES)
                rows.append({
                    "sample_id": f"{rec.circuit_id}|eps{rec.synthesis_epsilon:g}|p{p:g}|e{budget:g}",
                    "circuit_id": rec.circuit_id, "algorithm": rec.algorithm, "source": rec.source,
                    "abstraction_level": rec.abstraction_level, "num_qubits": logical_qubits,
                    "synthesis_epsilon": rec.synthesis_epsilon,
                    "physical_error_rate": p, "logical_error_budget": budget,
                    "code_distance": d, "rounds_total": rounds,
                    "logical_error_total": p_total,
                    "physical_qubits": int(physical),
                    "runtime_seconds": float(total_cycles * CYCLE_TIME),
                    "factory_count": n_factories,
                    "status": status,
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROC / "logical_resources.parquet"))
    ap.add_argument("--features", default=str(PROC / "circuit_features.parquet"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    params = {
        "physical_error_rate": [1e-4, 5e-4, 1e-3],
        "logical_error_budget": [1e-3, 1e-6, 1e-9],
        "code_cycle_time_seconds": CYCLE_TIME,
        "threshold": P_TH,
        "prefactor": PREFACTOR,
        "factory_patches": FACTORY_PATCHES,
        "model": "surface-code threshold scaling (analytic reference estimator)",
    }
    if args.config:
        params.update(json.loads(Path(args.config).read_text()))

    dataset = pd.read_parquet(args.dataset)
    features = pd.read_parquet(args.features)
    out = build(dataset, features, params)
    out.to_parquet(PROC / "fault_tolerant_resources.parquet", index=False)
    with open(PROC / "qec_model_config.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"records={len(out)} unique_circuits={out.circuit_id.nunique()} "
          f"unique_configs={out[['synthesis_epsilon','physical_error_rate','logical_error_budget']].drop_duplicates().shape[0]}")
    print(out.groupby("status").size().to_string())
    ok = out[out.status == "feasible"]
    if len(ok):
        print(f"code distance: {ok.code_distance.min()}–{ok.code_distance.max()}")
        print(f"physical qubits: {ok.physical_qubits.min():.3g}–{ok.physical_qubits.max():.3g}")
        print(f"runtime (s): {ok.runtime_seconds.min():.3g}–{ok.runtime_seconds.max():.3g}")


if __name__ == "__main__":
    main()
