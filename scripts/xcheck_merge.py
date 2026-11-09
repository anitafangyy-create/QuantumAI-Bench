#!/usr/bin/env python3
"""独立合成器交叉核验 · 第 3 步（本机）：把两端结果合并、出统计表。

输入 results/evaluation/xcheck_in.csv（本机 qiskit T 数）与 xcheck_out.csv（远端 pygridsynth），
输出 results/evaluation/xcheck_pairs.csv 与 xcheck_summary.json。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "evaluation"


def main() -> None:
    a = pd.read_csv(EVAL / "xcheck_in.csv")
    b = pd.read_csv(EVAL / "xcheck_out.csv")
    df = a.merge(b, on=["theta", "epsilon"], suffixes=("", "_remote"))
    df["ratio"] = df["t_pygridsynth"] / df["t_qiskit"]
    df.to_csv(EVAL / "xcheck_pairs.csv", index=False)

    res = {"n_pairs": int(len(df)), "n_theta": int(df.theta.nunique()), "by_eps": {}}
    for eps, g in df.groupby("epsilon"):
        k = f"{eps:g}"
        res["by_eps"][k] = {
            "n": int(len(g)),
            "t_qiskit_median": float(g.t_qiskit.median()),
            "t_pygridsynth_median": float(g.t_pygridsynth.median()),
            "exact_equal_frac": round(float((g.t_qiskit == g.t_pygridsynth).mean()), 4),
            "within_10pct_frac": round(float((g.ratio <= 1.10).mean()), 4),
            "ratio_median": round(float(g.ratio.median()), 3),
            "ratio_p10": round(float(g.ratio.quantile(0.10)), 3),
            "ratio_p90": round(float(g.ratio.quantile(0.90)), 3),
            "max_angle_err_remote": float(g.angle_err.max()),
            "frac_remote_err_le_eps": round(float((g.angle_err <= g.epsilon * 1.02).mean()), 4),
        }
    res["note"] = (
        "两端同属 Ross–Selinger 算法族但实现完全独立（qiskit Rust 后端 vs pygridsynth Python+mpmath）："
        "验证的是实现层面的一致性，而非算法层面的最优性；后者需要 Qualtran / Microsoft QRE，尚未接入。"
    )
    res["per_rotation_T_per_log2"] = {
        k: round(v["t_qiskit_median"] / (__import__("math").log2(1 / float(k))), 3)
        for k, v in res["by_eps"].items()
    }
    json.dump(res, open(EVAL / "xcheck_summary.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
