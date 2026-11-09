#!/usr/bin/env python3
"""统计检验：Cluster Bootstrap 置信区间 + Wilcoxon 配对检验 + Holm 多重比较校正。

为什么需要单独一个脚本：
  - metrics_raw.csv 只存聚合指标，无法做显著性判断。这里重新训练并按「线路」保留逐行误差，
    因为同一线路的 3 档 ε × 27 个配置是强相关的，必须以 circuit_id 为簇做重抽样，
    否则置信区间会被人为压窄（伪重复）。
  - 模型比较用配对 Wilcoxon（同一线路、同一配置下 RF vs GBDT 的绝对 log 误差），
    多重比较用 Holm 校正。

用法：
  .venv/bin/python scripts/stats_tests.py

产出：
  results/evaluation/stats_tests.csv    每个 (protocol, target, comparison) 的 CI 与 p 值
  results/evaluation/stats_bootstrap.csv  bootstrap 分布摘要
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "evaluation"


def load_tm():
    spec = importlib.util.spec_from_file_location("tm", ROOT / "scripts" / "train_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tm = load_tm()
EXTERNAL = tm.EXTERNAL_SOURCE


def per_circuit_errors(df: pd.DataFrame, protocol: str, target: str, models: list[str],
                       seeds: list[int]) -> pd.DataFrame:
    """训练并在测试集上返回逐行误差（保留 circuit_id 作为簇标记）。"""
    numeric_all = [c for c in tm.STRUCT_FEATURES + tm.CONFIG_FEATURES if c in df.columns]
    if target == "logical_qubits":
        numeric_all = [c for c in numeric_all if c != "num_qubits"]

    rows = []
    if protocol == "grouped":
        from sklearn.model_selection import GroupShuffleSplit
        sub = df.dropna(subset=[target]).copy()
        core = sub[sub["source"] != EXTERNAL]
        for seed in seeds:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
            tr_idx, te_idx = next(gss.split(core, groups=core["circuit_id"]))
            tr, te = core.iloc[tr_idx], core.iloc[te_idx]
            for model in models:
                pipe = tm.make_model(model, numeric_all, tm.CAT_FEATURES if model != "ridge" else [])
                cols = numeric_all + (tm.CAT_FEATURES if model != "ridge" else [])
                pipe.fit(tr[cols], np.log1p(tr[target]))
                pred = np.clip(np.expm1(pipe.predict(te[cols])), 0, None)
                rows.append(pd.DataFrame({
                    "seed": seed, "model": model, "circuit_id": te["circuit_id"].to_numpy(),
                    "y_true": te[target].to_numpy(dtype=float), "y_pred": pred,
                }))
    else:  # crosssrc
        sub = df.dropna(subset=[target]).copy()
        tr = sub[sub["source"] != EXTERNAL]
        te = sub[sub["source"] == EXTERNAL]
        if len(tr) < 50 or len(te) < 20:
            return pd.DataFrame()
        for model in models:
            pipe = tm.make_model(model, numeric_all, [])
            pipe.fit(tr[numeric_all], np.log1p(tr[target]))
            pred = np.clip(np.expm1(pipe.predict(te[numeric_all])), 0, None)
            rows.append(pd.DataFrame({
                "seed": -1, "model": model, "circuit_id": te["circuit_id"].to_numpy(),
                "y_true": te[target].to_numpy(dtype=float), "y_pred": pred,
            }))
    out = pd.concat(rows, ignore_index=True)
    valid = (out.y_true > 0) & (out.y_pred > 0)
    safe_true = out.y_true.where(out.y_true > 0, 1.0)
    safe_pred = out.y_pred.where(out.y_pred > 0, 1.0)
    out["abs_log10_err"] = np.where(valid, np.abs(np.log10(safe_pred / safe_true)), np.nan)
    return out


def circuit_level(err: pd.DataFrame) -> pd.DataFrame:
    """按 (seed, model, circuit_id) 聚合：取该线路所有配置的中位绝对 log10 误差。"""
    g = (err.dropna(subset=["abs_log10_err"])
         .groupby(["seed", "model", "circuit_id"])["abs_log10_err"].median().reset_index())
    return g


def cluster_bootstrap_ci(values: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float, float, bool]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), True
    boot = np.array([np.median(rng.choice(values, size=n, replace=True)) for _ in range(B)])
    med = float(np.median(values))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    # 中位数落在某个取值上的比例超过一半时，bootstrap 中位数会恒等于该值，CI 退化为一个点；
    # 这种情况必须显式标注，不能当成"高精度"报出去。
    atom = float((values == med).mean())
    return med, lo, hi, bool(atom > 0.5 or (hi - lo) == 0)


def cluster_bootstrap_prop(values: np.ndarray, thresh: float, B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """误差超过阈值的线路占比及其簇级 bootstrap 置信区间。

    比"中位数 CI"更稳：中位数在离散误差分布上会退化成点估计（bootstrap 中位数恒等），
    而比例是连续量，置信区间不会塌缩，适合在论文里报"有多少比例的线路误差超过 2 倍"。
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float((values > thresh).mean())
    boot = np.array([(rng.choice(values, size=n, replace=True) > thresh).mean() for _ in range(B)])
    return obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return list(adj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rf", "xgb"])
    ap.add_argument("--targets", nargs="+", default=["logical_t_count", "physical_qubits"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026, 2027, 2028])
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    df = tm.load_data()
    print(f"records={len(df)} circuits={df.circuit_id.nunique()}", flush=True)

    raw = []
    boot_rows = []
    tests = []
    fried_rows = []

    for protocol in ["grouped", "crosssrc"]:
        for target in args.targets:
            print(f"[{protocol}/{target}]", flush=True)
            err = per_circuit_errors(df, protocol, target, args.models, args.seeds)
            if not len(err):
                continue
            raw.append(err.assign(protocol=protocol, target=target))
            cl = circuit_level(err)

            # 每个模型的簇级 bootstrap 置信区间（中位绝对 log10 误差，换算成"倍"更直观）
            for model in args.models:
                vals = cl[cl.model == model]["abs_log10_err"].to_numpy()
                med, lo, hi, degenerate = cluster_bootstrap_ci(vals, B=args.bootstrap)
                q1, q3 = np.percentile(vals, [25, 75]) if len(vals) else (np.nan, np.nan)
                p2, p2lo, p2hi = cluster_bootstrap_prop(vals, np.log10(2.0), B=args.bootstrap)
                p10, p10lo, p10hi = cluster_bootstrap_prop(vals, 1.0, B=args.bootstrap)
                boot_rows.append({
                    "protocol": protocol, "target": target, "model": model,
                    "n_circuits": len(vals), "median_log10_err": med,
                    "ci_lo": lo, "ci_hi": hi, "ci_degenerate": degenerate,
                    "median_factor": float(10 ** med), "factor_lo": float(10 ** lo), "factor_hi": float(10 ** hi),
                    "iqr_factor_lo": float(10 ** q1), "iqr_factor_hi": float(10 ** q3),
                    "frac_gt2x": p2, "frac_gt2x_lo": p2lo, "frac_gt2x_hi": p2hi,
                    "frac_gt10x": p10, "frac_gt10x_lo": p10lo, "frac_gt10x_hi": p10hi,
                })

            # 三模型及以上：Friedman 整体检验（块 = 线路，同一个块上各模型的中位误差）
            if len(args.models) >= 3:
                wide3 = cl.pivot_table(index=["seed", "circuit_id"], columns="model", values="abs_log10_err").dropna()
                if len(wide3) >= 10:
                    stat, p = friedmanchisquare(*[wide3[m].to_numpy() for m in args.models])
                    ranks = wide3[args.models].rank(axis=1).mean()
                    fried_rows.append({
                        "protocol": protocol, "target": target,
                        "n_blocks": int(len(wide3)), "friedman_stat": float(stat), "p_raw": float(p),
                        "model_order": "; ".join(f"{m}={ranks[m]:.2f}" for m in args.models),
                        "best_by_mean_rank": str(ranks.idxmin()),
                    })

            # 配对比较：同一线路、同一 seed 下两个模型的误差差（多模型时两两对比）
            from itertools import combinations
            for a, b in combinations(args.models, 2):
                wide = cl.pivot_table(index=["seed", "circuit_id"], columns="model", values="abs_log10_err")
                wide = wide.dropna()
                if len(wide) >= 10:
                    diff = (wide[b] - wide[a]).to_numpy()
                    stat, p = wilcoxon(wide[a].to_numpy(), wide[b].to_numpy())
                    med_diff, lo, hi, degenerate = cluster_bootstrap_ci(diff, B=args.bootstrap)
                    tests.append({
                        "protocol": protocol, "target": target,
                        "comparison": f"{b} - {a}", "n_pairs": int(len(wide)),
                        "median_diff_log10": float(med_diff), "diff_ci_lo": lo, "diff_ci_hi": hi,
                        "diff_ci_degenerate": degenerate,
                        "better": b if med_diff < 0 else a,
                        "p_raw": float(p),
                    })

    if not tests and not fried_rows:
        print("没有可比对的样本")
        return

    fried_df = pd.DataFrame(fried_rows)
    if len(fried_df):
        fried_df["p_holm"] = holm(fried_df["p_raw"].tolist())
        fried_df["significant_after_holm"] = fried_df["p_holm"] < 0.05
        fried_df.to_csv(EVAL / "stats_friedman.csv", index=False)
        print("\nFriedman 整体检验（块 = 线路，平均秩越小越好）：")
        print(fried_df[["protocol", "target", "n_blocks", "friedman_stat", "p_raw", "p_holm",
                        "best_by_mean_rank", "model_order"]].round(4).to_string(index=False))

    if tests:
        tests_df = pd.DataFrame(tests)
        tests_df["p_holm"] = holm(tests_df["p_raw"].tolist())
        tests_df["significant_after_holm"] = tests_df["p_holm"] < 0.05
        tests_df.to_csv(EVAL / "stats_tests.csv", index=False)
    else:
        tests_df = pd.DataFrame()

    EVAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(boot_rows).to_csv(EVAL / "stats_bootstrap.csv", index=False)
    pd.concat(raw, ignore_index=True).to_csv(EVAL / "per_row_errors.csv", index=False)

    with open(EVAL / "stats_tests.json", "w") as f:
        json.dump({
            "n_bootstrap": args.bootstrap,
            "cluster_unit": "circuit_id",
            "test": "wilcoxon signed-rank on circuit-level median absolute log10 error",
            "multiplicity": "Holm",
            "note": "ci_degenerate=True 表示中位数落在占比>50%的取值上，bootstrap CI 退化为点，不能当作高精度",
            "results": tests_df.to_dict(orient="records") if len(tests_df) else [],
            "friedman": fried_rows,
            "bootstrap": boot_rows,
        }, f, ensure_ascii=False, indent=2)

    if len(tests_df):
        print(tests_df[["protocol", "target", "comparison", "n_pairs", "median_diff_log10",
                        "p_raw", "p_holm", "significant_after_holm", "better"]].round(4).to_string(index=False))
    print("\n簇级 bootstrap（中位绝对误差 factor = 10^median，>1 即平均偏离倍数）：")
    print(pd.DataFrame(boot_rows)[["protocol", "target", "model", "n_circuits", "median_factor",
                                   "factor_lo", "factor_hi", "ci_degenerate"]].round(3).to_string(index=False))
    print("\n误差比例（含簇级 bootstrap 95% CI，连续量，不受离散中位数塌缩影响）：")
    print(pd.DataFrame(boot_rows)[["protocol", "target", "model", "frac_gt2x", "frac_gt2x_lo",
                                   "frac_gt2x_hi", "frac_gt10x", "frac_gt10x_lo", "frac_gt10x_hi"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
