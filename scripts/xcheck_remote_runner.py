#!/usr/bin/env python3
"""独立合成器交叉核验 · 第 2 步（远端，Linux）：用 pygridsynth 对同一批 (θ, ε) 独立合成。

为什么必须在 Linux 上跑：pyLIQTR/pygridsynth 依赖 numba→llvmlite，macOS 上无预编译 wheel，
会退化成源码编译 LLVM 并失败。远端环境的搭法见 docs/XCHECK_INDEPENDENT_SYNTHESIS.md。

用法（远端）: python xcheck_remote_runner.py in.csv out.csv
"""
from __future__ import annotations

import csv
import cmath
import math
import sys
import warnings

import numpy as np
import pygridsynth

warnings.filterwarnings("ignore")

H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
T = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)
S = np.diag([1, 1j]).astype(complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
# pygridsynth 的 W 门是全局相位门（exp(iπ/8)·I），不计 T
W = np.eye(2, dtype=complex) * cmath.exp(1j * math.pi / 8)
GATE = {"H": H, "T": T, "S": S, "X": X, "W": W}


def unitary(seq: str) -> np.ndarray:
    u = np.eye(2, dtype=complex)
    for ch in seq:
        if ch in GATE:
            u = GATE[ch] @ u
    return u


def angle_of(u: np.ndarray) -> float:
    phi = cmath.phase(np.linalg.det(u)) / 2
    r = u * cmath.exp(-1j * phi)
    return 2 * math.atan2(r[1, 1].imag, r[1, 1].real)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "in.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "out.csv"
    rows = list(csv.DictReader(open(src)))
    out = []
    for r in rows:
        th, eps, tq = float(r["theta"]), float(r["epsilon"]), int(r["t_qiskit"])
        s = pygridsynth.gridsynth_gates(th, eps)
        err = abs(angle_of(unitary(s)) - th)
        err = min(err, abs(2 * math.pi - err))
        out.append({"theta": th, "epsilon": eps, "t_qiskit": tq,
                    "t_pygridsynth": s.count("T"), "angle_err": err, "n_gates": len(s)})
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"done rows={len(out)} -> {dst}")


if __name__ == "__main__":
    main()
