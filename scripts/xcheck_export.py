#!/usr/bin/env python3
"""独立合成器交叉核验 · 第 1 步（本机）：生成角度集并算出"本仓库实现"的 T 门数 + 实测 ε 合规。

产出的 xcheck_in.csv 会被送到远端（Linux + pygridsynth）跑一遍独立实现，
再用 scripts/xcheck_merge.py 对账。角度集用固定种子生成，保证可复现。
"""
from __future__ import annotations

import argparse
import cmath
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from qiskit.synthesis.discrete_basis.ross_selinger import gridsynth_rz

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "evaluation"

EPS = [1e-3, 1e-6, 1e-9]
MAT = {
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    "t": np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex),
    "s": np.diag([1, 1j]).astype(complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
}


def unitary(qc) -> np.ndarray:
    u = np.eye(2, dtype=complex)
    for inst in qc.data:
        name = inst.operation.name
        if name == "tdg":
            u = np.conj(MAT["t"].T) @ u
        elif name == "sdg":
            u = np.conj(MAT["s"].T) @ u
        elif name == "z":
            u = np.diag([1, -1]).astype(complex) @ u
        elif name in MAT:
            u = MAT[name] @ u
    return u


def angle_of(u: np.ndarray) -> float:
    phi = cmath.phase(np.linalg.det(u)) / 2
    r = u * cmath.exp(-1j * phi)
    return 2 * math.atan2(r[1, 1].imag, r[1, 1].real)


def build_thetas(n_random: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    thetas = [rng.uniform(-math.pi, math.pi) for _ in range(n_random)]
    for m in range(2, 12):  # QFT / QPE 型 kπ/2^m
        for k in range(1, 2 ** min(m, 6)):
            a = k * math.pi / (2 ** m)
            kk = a / (math.pi / 4)
            if abs(kk - round(kk)) > 1e-12:
                thetas.append(a)
    return sorted({round(t, 12) for t in thetas})[:500]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-thetas", type=int, default=500)
    args = ap.parse_args()

    thetas = build_thetas(args.n_random, args.seed)[: args.max_thetas]
    rows, audit = [], []
    for th in thetas:
        for eps in EPS:
            qc = gridsynth_rz(th, epsilon=eps)
            t = max(int(qc.count_ops().get("t", 0)), 1)
            err = abs(angle_of(unitary(qc)) - th)
            err = min(err, abs(2 * math.pi - err))
            rows.append({"theta": th, "epsilon": eps, "t_qiskit": t})
            audit.append({"theta": th, "epsilon": eps, "t_qiskit": t,
                          "angle_err": err, "err_over_eps": err / eps,
                          "violates_eps": err > eps * 1.02})

    with open(EVAL / "xcheck_in.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["theta", "epsilon", "t_qiskit"])
        w.writeheader()
        w.writerows(rows)

    import pandas as pd
    a = pd.DataFrame(audit)
    a.to_csv(EVAL / "qiskit_eps_audit.csv", index=False)
    summary = {
        "n_thetas": len(thetas), "n_pairs": len(rows),
        "violations": int(a.violates_eps.sum()),
        "err_over_eps_median": round(float(a.err_over_eps.median()), 4),
        "err_over_eps_p90": round(float(a.err_over_eps.quantile(0.9)), 4),
        "t_median_by_eps": {f"{k:g}": float(v) for k, v in a.groupby("epsilon").t_qiskit.median().items()},
    }
    json.dump(summary, open(EVAL / "qiskit_eps_audit.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
