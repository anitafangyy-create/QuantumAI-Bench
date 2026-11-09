#!/usr/bin/env python3
"""导出"角度级"交叉核验样本。

为什么要角度级而不是线路级：独立估计器（pyLIQTR/pygridsynth）只需要 (θ, ε) 就能给 T 门数，
传角度比传 6,938 个 qasm 便宜几个数量级，且能逐点对账，比整条线路的总数更有诊断力。

产出：results/evaluation/xcheck_angles.csv —— 每行一个去重后的 (θ, ε)，
含本仓库 `build_dataset.clifford_t_decompose` 同一套 gridsynth 给出的 T 门数 t_ours。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from qiskit import qasm2, transpile
from qiskit.synthesis.discrete_basis.ross_selinger import gridsynth_rz

ROOT = Path(__file__).resolve().parents[1]
EPS = [1e-3, 1e-6, 1e-9]


def angle_class(theta: float) -> tuple[bool, int]:
    """与 build_dataset.py 完全一致：Clifford 角（kπ/4）判定 + T 数。"""
    import math
    k = theta / (math.pi / 4)
    if abs(k - round(k)) < 1e-12:
        k_int = int(round(k)) % 8
        return True, 0 if k_int % 2 == 0 else 1
    return False, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-circuits", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="results/evaluation/xcheck_angles.csv")
    args = ap.parse_args()

    lab = pd.read_parquet(ROOT / "processed_data" / "logical_resources.parquet")
    feats = pd.read_parquet(ROOT / "processed_data" / "circuit_features.parquet")
    man = pd.read_parquet(ROOT / "raw_data" / "circuits_manifest.parquet")

    # 按 T-count 分 10 档分层抽样，避免只抽到小线路
    t = lab[lab.synthesis_epsilon == 1e-6].set_index("circuit_id")["logical_t_count"].sort_values()
    rng = random.Random(args.seed)
    buckets = [list(t.index)[i::10] for i in range(10)]
    pick = []
    per = max(args.n_circuits // 10, 1)
    for b in buckets:
        pick += rng.sample(b, min(per, len(b)))
    pick = list(dict.fromkeys(pick))
    print(f"sampled {len(pick)} circuits across 10 T-count deciles")

    rows = []
    for cid in pick:
        sub = man[man.circuit_id == cid]
        if sub.empty or sub.iloc[0].qasm_status != "ok":
            continue
        try:
            qc = qasm2.loads(Path(sub.iloc[0].qasm_path).read_text())
            base = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0, seed_transpiler=0)
        except Exception:
            continue
        for inst in base.data:
            if inst.operation.name != "rz":
                continue
            theta = float(inst.operation.params[0])
            is_cliff, tc = angle_class(theta)
            if is_cliff:
                continue
            rows.append({"circuit_id": cid, "theta": round(theta, 12)})

    df = pd.DataFrame(rows).drop_duplicates()
    print(f"unique non-Clifford (circuit, theta) pairs: {len(df)}")

    out = []
    for _, r in df.iterrows():
        th = float(r.theta)
        for eps in EPS:
            try:
                t_ours = max(int(gridsynth_rz(th, epsilon=eps).count_ops().get("t", 0)), 1)
            except Exception:
                continue
            out.append({"circuit_id": r.circuit_id, "theta": th, "epsilon": eps, "t_ours": t_ours})
    res = pd.DataFrame(out)
    path = ROOT / args.out
    res.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(res)}  unique_theta={res.theta.nunique()}")
    print(res.groupby("epsilon").t_ours.describe()[["count", "50%", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
