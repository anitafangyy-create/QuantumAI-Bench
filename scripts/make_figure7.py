#!/usr/bin/env python3
"""图 7：把 §4.5 的误差比例 + 95% CI 画出来。

为什么单独一张图：中位偏离在跨源会退化成点估计（ci_degenerate），而误差比例是连续量，
它的 CI 才是"跨源到底有多不可靠"的可视化证据。左栏 grouped（对照），右栏 crosssrc。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "evaluation"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_COLOR = {"ridge": "#8c564b", "rf": "#1f77b4", "xgb": "#2ca02c"}
MODEL_LABEL = {"ridge": "Ridge", "rf": "RandomForest", "xgb": "GBDT"}


def draw(ax, df, target_label):
    order = ["ridge", "rf", "xgb"]
    span = max(float(df["frac_gt2x"].max()) * 100 * 1.45, 8.0) if not df.empty else 10.0
    n = len(order)
    ys = np.arange(n)[::-1]
    for y, m in zip(ys, order):
        r = df[df.model == m]
        if r.empty:
            continue
        v = float(r["frac_gt2x"].iloc[0]) * 100
        lo = float(r["frac_gt2x_lo"].iloc[0]) * 100
        hi = float(r["frac_gt2x_hi"].iloc[0]) * 100
        ax.barh(y, v, color=MODEL_COLOR[m], height=0.55)
        ax.errorbar(v, y, xerr=[[max(v - lo, 0)], [max(hi - v, 0)]],
                    fmt="none", ecolor="black", capsize=3, lw=1)
        ax.text(v + span * 0.02, y, f"{v:.1f}%  [{lo:.1f}, {hi:.1f}]", va="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([MODEL_LABEL[m] for m in order])
    ax.set_xlabel("Share off by >2x (%), 95% CI", fontsize=8.5, labelpad=4)
    ax.set_title(target_label, fontsize=11)
    ax.grid(axis="x", alpha=0.3)


def main() -> None:
    bs = pd.read_csv(EVAL / "stats_bootstrap.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0))
    combos = [("grouped", "logical_t_count", "(a) grouped, logical T-count"),
              ("grouped", "physical_qubits", "(b) grouped, physical qubits"),
              ("crosssrc", "logical_t_count", "(c) cross-source, logical T-count"),
              ("crosssrc", "physical_qubits", "(d) cross-source, physical qubits")]
    for ax, (proto, target, title) in zip(axes.flat, combos):
        d = bs[(bs.protocol == proto) & (bs.target == target)]
        draw(ax, d, title)
        _mx = 0.0 if d.empty else float(d["frac_gt2x"].max()) * 100
        ax.set_xlim(0, max(_mx * 1.5, 8.0))

    _cs = bs[(bs.protocol == "crosssrc") & (bs.model.isin(["rf", "xgb"]))]
    _t = _cs[_cs.target == "logical_t_count"]
    _q = _cs[_cs.target == "physical_qubits"]
    _lo, _hi = float(_t["median_factor"].min()), float(_t["median_factor"].max())
    _qlo, _qhi = float(_q["median_factor"].min()), float(_q["median_factor"].max())
    _plo, _phi = float(_cs["frac_gt2x"].min()) * 100, float(_cs["frac_gt2x"].max()) * 100
    fig.suptitle("Error fractions with cluster-bootstrap CIs (cluster = circuit_id, B = 2000)", fontsize=12.5)
    fig.text(0.01, 0.012,
             "Bar = share of circuits whose median |log10 error| exceeds log10(2); whiskers = 95% cluster bootstrap CI.\n"
             f"Cross-source, RF/GBDT only: T-count deviation {_lo:.2f}-{_hi:.2f}x, physical {_qlo:.2f}-{_qhi:.2f}x, "
             f"but {_plo:.1f}-{_phi:.1f}% of circuits are off by >2x - median and tail must both be reported.",
             fontsize=7.6, color="#444444", linespacing=1.4)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94], h_pad=2.2, w_pad=2.0)
    out = FIG / "fig7_error_fraction_ci.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
