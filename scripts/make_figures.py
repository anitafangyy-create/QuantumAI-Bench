#!/usr/bin/env python3
"""阶段四：结果图表（论文 Figure 1–4 的可复现版本）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
EVAL = ROOT / "results" / "evaluation"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def fig_tcount_scaling(feat: pd.DataFrame, lab: pd.DataFrame) -> None:
    """T-count 随规模的增长：按算法族展示不同的标度指数。"""
    eps = lab.synthesis_epsilon.min()
    lab = lab[(lab.synthesis_epsilon == eps) & (lab.abstraction_level == "INDEP")]
    top = lab.groupby("algorithm")["circuit_id"].nunique().sort_values(ascending=False).head(14).index
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cmap = plt.get_cmap("tab20")
    k = 0
    for algo in sorted(top):
        g = lab[lab.algorithm == algo].groupby("num_qubits")["logical_t_count"].median().sort_index()
        if len(g) < 2 or g.max() <= 0:      # 全 Clifford 算法 T-count 恒为 0，log 轴无法显示
            continue
        ax.plot(g.index, g.values, marker="o", ms=3, lw=1.2, color=cmap(k % 20), label=algo)
        k += 1
    ax.set_xscale("linear"); ax.set_yscale("log")
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32])
    ax.set_xlabel("Number of logical qubits n"); ax.set_ylabel(f"Logical T-count (ε={eps:g})")
    ax.set_title("T-count scaling across algorithm families (MQT Bench INDEP level)")
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.text(0.01, -0.02, "All-Clifford families (bv / dj / ghz / graphstate, T-count ≡ 0) are omitted on this log axis.",
             fontsize=6.5, color="#555555")
    fig.tight_layout(); fig.savefig(FIG / "fig1_tcount_scaling.png", bbox_inches="tight"); plt.close(fig)


def fig_qec_tradeoff(ft: pd.DataFrame, lab: pd.DataFrame) -> None:
    """物理量子比特需求随算法 T-count 的增长，按 (p, budget) 分族画中位趋势线。"""
    sub = ft[ft.status == "feasible"].merge(
        lab[["circuit_id", "synthesis_epsilon", "logical_t_count"]],
        on=["circuit_id", "synthesis_epsilon"], how="left")
    sub = sub[sub.logical_t_count > 0]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cmap = plt.get_cmap("viridis")
    keys = [(p, b) for p in sorted(sub.physical_error_rate.unique())
            for b in sorted(sub.logical_error_budget.unique())]
    xbins = np.logspace(np.log10(sub.logical_t_count.min()), np.log10(sub.logical_t_count.max()), 14)
    for i, key in enumerate(keys):
        g = sub[(sub.physical_error_rate == key[0]) & (sub.logical_error_budget == key[1])]
        if not len(g):
            continue
        mid = (xbins[:-1] * xbins[1:]) ** 0.5
        med = []
        xs = []
        for j, (lo, hi) in enumerate(zip(xbins[:-1], xbins[1:])):
            cell = g[(g.logical_t_count >= lo) & (g.logical_t_count < hi)]
            if len(cell) >= 5:            # 少于 5 条线路的 bin 中位数不可信，直接丢弃以免出现虚假拐点
                xs.append(mid[j]); med.append(cell.physical_qubits.median())
        ax.plot(xs, med, marker="o", ms=3, lw=1.3, color=cmap(i / max(1, len(keys) - 1)),
                label=f"p={key[0]:g}, ε_b={key[1]:g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Logical T-count of the circuit"); ax.set_ylabel("Median physical qubits")
    ax.set_title("Physical resource growth vs algorithm T-count, by QEC configuration")
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="upper left")
    fig.text(0.01, -0.06,
             "Each point is the median over >=5 circuits in a log-spaced T-count bin. Steps come from the integer code\n"
             "distance d. The dip between 1e4 and 1e5 T-count is bin composition, not noise: the median magic-state\n"
             "factory count falls from 7 to 2 (shorter T-depth circuits need fewer factories), which shrinks the\n"
             "physical footprint faster than the rising d adds to it.",
             fontsize=6.3, color="#555555", va="top")
    fig.tight_layout(); fig.savefig(FIG / "fig2_qec_tradeoff.png", bbox_inches="tight"); plt.close(fig)


def fig_pred_vs_true() -> None:
    path = EVAL / "predictions_grouped_xgb.csv"
    if not path.exists():
        return
    pred = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, target in zip(axes, ["logical_t_count", "physical_qubits", "runtime_seconds"]):
        g = pred[pred.target == target]
        if not len(g):
            continue
        ax.scatter(g.y_true, g.y_pred, s=5, alpha=0.4, color="#3b6ea5")
        lo, hi = max(g.y_true.min(), 1), max(g.y_true.max(), g.y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Reference estimator"); ax.set_ylabel("GBDT prediction")
        ax.set_title(target)
    fig.suptitle("Predicted vs analytic reference resources (grouped split, GBDT)")
    fig.tight_layout(); fig.savefig(FIG / "fig3_pred_vs_true.png"); plt.close(fig)


def fig_loso() -> None:
    path = EVAL / "metrics_raw.csv"
    if not path.exists():
        return
    m = pd.read_csv(path)
    m = m[(m.protocol == "loso") & (m.model == "xgb") & (m.target == "logical_t_count")]
    if not len(m):
        return
    m = m.dropna(subset=["mdape_pct"])
    m = m.groupby("heldout")["mdape_pct"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.2, max(2.6, 0.24 * len(m))))
    ax.barh(m.index, m.values, color="#c46a4b")
    ax.set_xlabel("Median absolute percentage error (%)")
    ax.set_title("Leave-one-algorithm-out generalisation (target: logical T-count)", fontsize=9)
    fig.text(0.01, -0.01, "4 all-Clifford algorithms (bv / dj / ghz / graphstate) have T-count = 0; "
                         "their relative error is undefined and is excluded here.",
             fontsize=6.5, color="#555555")
    fig.tight_layout(); fig.savefig(FIG / "fig4_loso.png", bbox_inches="tight"); plt.close(fig)


def fig_scale_extrapolation() -> None:
    path = EVAL / "metrics_raw.csv"
    if not path.exists():
        return
    m = pd.read_csv(path)
    m = m[(m.protocol == "scale") & (m.target == "logical_t_count")]
    if not len(m):
        return
    agg = m.groupby("model")[["r2_log", "mdape_pct"]].mean()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    order = ["ridge", "rf", "xgb"]
    agg = agg.reindex([m for m in order if m in agg.index])

    x = np.arange(len(agg))
    clipped = agg.r2_log.clip(lower=-2)
    ax.bar(x, clipped.values, width=0.55, color="#3b6ea5")
    for i, (raw, clipped_v) in enumerate(zip(agg.r2_log.values, clipped.values)):
        ax.text(i, max(clipped_v, -2) + 0.05, f"{raw:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(-2.2, 1.15)
    ax.set_xticks(x); ax.set_xticklabels(agg.index)
    ax.set_ylabel("R² (log space)" + ("  (clipped at −2)" if agg.r2_log.min() < -2 else ""))
    ax.set_title("Extrapolation to >16-qubit circuits\n(target: logical T-count)", fontsize=9)

    ax2.bar(x, agg.mdape_pct.values, width=0.55, color="#c46a4b")
    for i, v in enumerate(agg.mdape_pct.values):
        ax2.text(i, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x); ax2.set_xticklabels(agg.index)
    ax2.set_yscale("log"); ax2.set_ylabel("MdAPE (%), log scale")
    ax2.set_title("Same models, relative error\n(non-zero references only)", fontsize=9)

    worst = agg.r2_log.min()
    note = (f"Bars clipped at −2 for readability; lowest raw R² = {worst:.2f} (linear model diverges outside the training scale range)."
            if worst < -2 else
            f"No bar needed clipping; lowest raw R² = {worst:.2f} (linear model collapses outside the training scale range).")
    fig.text(0.01, -0.03, note, fontsize=6.5, color="#555555")
    fig.tight_layout(); fig.savefig(FIG / "fig5_scale_extrapolation.png", bbox_inches="tight"); plt.close(fig)


def fig_cross_source() -> None:
    """同分布（MQT Bench 内）与跨源（MQT Bench → QCircuitBench）的误差对比。"""
    path = EVAL / "metrics_raw.csv"
    if not path.exists():
        return
    m = pd.read_csv(path)
    need = m[m.protocol.isin(["grouped", "crosssrc"])]
    if not len(need):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, target in zip(axes, ["logical_t_count", "physical_qubits"]):
        g = need[need.target == target]
        if not len(g):
            continue
        piv = g.pivot_table(index="model", columns="protocol", values="mdape_pct", aggfunc="mean")
        order = [x for x in ["ridge", "rf", "xgb"] if x in piv.index]
        piv = piv.reindex(order)
        x = np.arange(len(piv))
        w = 0.36
        ax.bar(x - w / 2, piv.get("grouped", pd.Series(index=piv.index, dtype=float)).values,
               width=w, color="#3b6ea5", label="in-distribution (grouped)")
        ax.bar(x + w / 2, piv.get("crosssrc", pd.Series(index=piv.index, dtype=float)).values,
               width=w, color="#c46a4b", label="cross-source (→ QCircuitBench)")
        for i, v in enumerate(piv.get("crosssrc", pd.Series(dtype=float)).values):
            ax.text(i + w / 2, v * 1.05, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(piv.index)
        ax.set_ylabel("MdAPE (%), log scale")
        ax.set_title(f"{target}", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle("Cross-source transfer: training on MQT Bench, testing on QCircuitBench", fontsize=9)
    fig.text(0.01, -0.03, "QCircuitBench never enters training for protocols A/B/C. "
                          "Its Algorithm_Design / benchmark_subset circuits are OpenQASM 3.0; imported via recursive include inlining.",
             fontsize=6.3, color="#555555")
    fig.tight_layout(); fig.savefig(FIG / "fig6_cross_source.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    feat = pd.read_parquet(PROC / "circuit_features.parquet")
    lab = pd.read_parquet(PROC / "logical_resources.parquet")
    ft = pd.read_parquet(PROC / "fault_tolerant_resources.parquet")
    fig_tcount_scaling(feat, lab)
    fig_qec_tradeoff(ft, lab)
    fig_pred_vs_true()
    fig_loso()
    fig_scale_extrapolation()
    fig_cross_source()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
