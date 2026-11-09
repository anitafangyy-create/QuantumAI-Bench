#!/usr/bin/env python3
"""阶段三：基线模型与泛化评估。

三类评估协议（避免同源线路泄漏）：
  A. grouped   —— 按 circuit_id 分组 70/15/15，同一线路的多个配置只出现在一个分区
  B. loso      —— Leave-One-Algorithm-Out：训练集不含该算法，测试只用该算法
  C. scale     —— 训练 ≤ N_MAX 量子比特，测试更大规模

特征集：
  F_struct : 线路结构/拓扑 + 配置参数（不含算法身份）
  F_algo   : F_struct + 算法 one-hot（仅在分组协议中使用，会出现类别缺失问题，故 LOSO 只用 F_struct）

指标：log 空间与原始空间的 MAE / RMSE / R² / MdAPE / sMAPE
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # macOS 上 xgboost 需要 libomp，缺失时自动降级到同族 GBDT，保证流水线可跑
    from xgboost import XGBRegressor
    _XGB_OK = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingRegressor
    XGBRegressor = None
    _XGB_OK = False

XGB_BACKEND = "xgboost" if _XGB_OK else "sklearn_hist_gbdt"

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
EVAL = ROOT / "results" / "evaluation"

TARGETS = ["logical_t_count", "logical_t_depth", "logical_qubits", "physical_qubits", "runtime_seconds"]

STRUCT_FEATURES = [
    "num_qubits", "gate_count", "circuit_depth", "one_qubit_gates", "two_qubit_gates",
    "multi_qubit_gates", "cx_count", "t_count_orig", "tdg_count_orig", "ccx_count_orig",
    "rz_count_orig", "distinct_gate_types", "gate_type_entropy", "gate_density",
    "gates_per_layer", "depth_per_qubit", "interaction_edges", "degree_mean", "degree_max",
    "graph_density", "connected_components", "largest_component",
    "n_clifford_rz", "n_nonclifford_rz", "t_count_clifford_part", "t_depth_structural",
    "clifford_ops_after_ct", "cx_after_ct",
]
CONFIG_FEATURES = ["log_epsilon", "log_p", "log_budget"]
CAT_FEATURES = ["algorithm"]
LOSO_TARGETS = ("logical_t_count", "physical_qubits")
EXTERNAL_SOURCE = "qcircuitbench"      # 跨源协议的测试集来源（训练集为其余来源，即 MQT Bench）


def load_data() -> pd.DataFrame:
    """三层数据合并：逻辑资源标签 + 物理/QEC 标签 + 线路结构特征。

    logical_resources 与 fault_tolerant_resources 按 (circuit_id, synthesis_epsilon) 一对多对齐，
    再按 circuit_id 挂上结构特征；任何一层缺失的记录直接丢弃（不做零填充）。
    """
    ft = pd.read_parquet(PROC / "fault_tolerant_resources.parquet")
    lab = pd.read_parquet(PROC / "logical_resources.parquet")[
        ["circuit_id", "synthesis_epsilon", "logical_t_count", "logical_t_depth", "logical_qubits"]
    ]
    feat = pd.read_parquet(PROC / "circuit_features.parquet")
    keep = ["circuit_id"] + [c for c in STRUCT_FEATURES if c in feat.columns]

    df = ft.merge(lab, on=["circuit_id", "synthesis_epsilon"], how="inner")
    df = df.merge(feat[keep].drop_duplicates("circuit_id"), on="circuit_id", how="left", suffixes=("", "_feat"))
    df["log_epsilon"] = np.log10(df["synthesis_epsilon"])
    df["log_p"] = np.log10(df["physical_error_rate"])
    df["log_budget"] = np.log10(df["logical_error_budget"])
    df = df[df["status"] == "feasible"].copy()
    for t in TARGETS:
        df[t] = pd.to_numeric(df[t], errors="coerce")
    df = df.dropna(subset=["logical_t_count", "logical_t_depth", "logical_qubits", "physical_qubits", "runtime_seconds"])
    return df


def make_model(name: str, numeric: list[str], categorical: list[str]):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], remainder="drop")
    if name == "ridge":
        est = Ridge(alpha=1.0)
    elif name == "rf":
        est = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=0)
    elif name == "xgb":
        if _XGB_OK:
            est = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08,
                               subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                               n_jobs=-1, tree_method="hist", random_state=0)
        else:
            est = HistGradientBoostingRegressor(max_iter=400, max_depth=6, learning_rate=0.08,
                                                l2_regularization=1.0, random_state=0)
    elif name == "mlp":
        est = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=True,
                           learning_rate_init=1e-3, random_state=0)
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("est", est)])


def metrics(y_true: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """y_true 为原始尺度；内部在 log1p 空间和原始空间分别评估。"""
    y_true_log = np.log1p(y_true)
    # 线性模型在跨源远距离样本上会给出极端 log 预测，expm1 直接溢出成 inf 并让 MAE 变成 nan/inf。
    # 先把 log 预测钳到 [−50, 20]（对应 1e20，已远超语料最大 3.86e6），再做非有限值兜底。
    y_pred_log = np.clip(np.asarray(y_pred_log, dtype=float), -50.0, 20.0)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)
    y_max = float(np.max(y_true)) if len(y_true) else 1.0
    y_pred = np.where(np.isfinite(y_pred), y_pred, max(y_max * 10.0, 1.0))
    eps = 1e-9
    pos = y_true > 0                      # 相对误差只在非零参考值上定义，否则分母被 eps 支配会出现 1e60% 这类无意义数值
    if pos.sum():
        log10_ratio = np.abs(np.log10(np.maximum(y_pred[pos] / y_true[pos], 1e-300)))
        # 相对误差封顶 1e6%，并单独报告"偏离 >10 倍"的比例（外推崩溃率）
        mdape = float(np.median(np.minimum(log10_ratio * 100, 1e6)))
        frac_gt_10x = float(np.mean(log10_ratio > 1.0))
    else:
        mdape, frac_gt_10x = float("nan"), float("nan")
    smape = np.mean(2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_pred) + np.abs(y_true), eps)) * 100
    return {
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "mae_orig": float(mean_absolute_error(y_true, y_pred)),
        "rmse_orig": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mdape_pct": mdape,
        "frac_gt_10x": frac_gt_10x,
        "smape_pct": float(smape),
        "n": int(len(y_true)),
        "n_nonzero_ref": int(pos.sum()),
    }


def run_protocol(df: pd.DataFrame, protocol: str, targets: list[str], models: list[str],
                 seeds: list[int], n_qubits_threshold: int = 16,
                 pred_store: list[pd.DataFrame] | None = None,
                 drop_zero_train: set[str] | None = None) -> pd.DataFrame:
    """drop_zero_train：只在**训练侧**剔除的逻辑 T-count 恒为 0 的线路 id 集合。

    这些线路全是 Clifford（bv/dj/ghz/graphstate 等），回归目标退化为常数 0，对 T-count 学习无信息，
    却会把"多比特 ⇒ 零 T"的先验灌进模型（见报告 §4.9）。测试集保持原样，保证与剔除前指标可比。
    """
    zs = drop_zero_train or set()
    rows = []
    for target in targets:
        sub = df.dropna(subset=[target]).copy()
        if len(sub) < 50:
            continue
        # LOSO 是最贵的协议（每个算法都要重训一次），只保留主目标，其余目标在 grouped/scale 中覆盖
        if protocol == "loso" and target not in LOSO_TARGETS:
            continue
        print(f"   {protocol}/{target}", flush=True)
        y = sub[target].to_numpy(dtype=float)
        for model_name in models:
            if protocol == "grouped":
                for seed in seeds:
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
                    tr_idx, te_idx = next(gss.split(sub, groups=sub["circuit_id"]))
                    tr, te = sub.iloc[tr_idx], sub.iloc[te_idx]
                    if zs:
                        tr = tr[~tr.circuit_id.isin(zs)]
                    numeric = [c for c in STRUCT_FEATURES + CONFIG_FEATURES if c in sub.columns]
                    if target == "logical_qubits":
                        # logical_qubits == num_qubits，若把 num_qubits 当特征就是恒等映射（R²=1、MAE≈0），
                        # 因此该目标必须屏蔽 num_qubits，只从线路结构推断量子比特规模。
                        numeric = [c for c in numeric if c != "num_qubits"]
                    cat = CAT_FEATURES if model_name != "ridge" else []
                    pipe = make_model(model_name, numeric, cat)
                    pipe.fit(tr[numeric + cat], np.log1p(tr[target]))
                    pred = pipe.predict(te[numeric + cat])
                    m = metrics(te[target].to_numpy(dtype=float), pred)
                    m.update({"target": target, "model": model_name, "protocol": "grouped", "seed": seed})
                    rows.append(m)
                    if pred_store is not None and model_name == "xgb" and seed == seeds[0]:
                        pred_store.append(pd.DataFrame({
                            "target": target, "circuit_id": te["circuit_id"].to_numpy(),
                            "algorithm": te["algorithm"].to_numpy(), "num_qubits": te["num_qubits"].to_numpy(),
                            "y_true": te[target].to_numpy(dtype=float),
                            "y_pred": np.clip(np.expm1(pred), 0, None),
                        }))
            elif protocol == "loso":
                numeric = [c for c in STRUCT_FEATURES + CONFIG_FEATURES if c in sub.columns]
                for algo in sorted(sub["algorithm"].unique()):
                    if (sub["algorithm"] == algo).sum() < 20:
                        continue
                    tr = sub[sub["algorithm"] != algo]
                    if zs:
                        tr = tr[~tr.circuit_id.isin(zs)]
                    te = sub[sub["algorithm"] == algo]
                    if tr["circuit_id"].nunique() < 10:
                        continue
                    pipe = make_model(model_name, numeric, [])
                    pipe.fit(tr[numeric], np.log1p(tr[target]))
                    pred = pipe.predict(te[numeric])
                    m = metrics(te[target].to_numpy(dtype=float), pred)
                    m.update({"target": target, "model": model_name, "protocol": "loso", "heldout": algo})
                    rows.append(m)
            elif protocol == "scale":
                numeric = [c for c in STRUCT_FEATURES + CONFIG_FEATURES if c in sub.columns]
                tr_mask = sub["num_qubits"] <= n_qubits_threshold
                if zs:
                    tr_mask = tr_mask & ~sub["circuit_id"].isin(zs)
                te_mask = sub["num_qubits"] > n_qubits_threshold
                if tr_mask.sum() < 50 or te_mask.sum() < 20:
                    continue
                pipe = make_model(model_name, numeric, [])
                pipe.fit(sub.loc[tr_mask, numeric], np.log1p(sub.loc[tr_mask, target]))
                pred = pipe.predict(sub.loc[te_mask, numeric])
                m = metrics(sub.loc[te_mask, target].to_numpy(dtype=float), pred)
                m.update({"target": target, "model": model_name, "protocol": "scale",
                          "train_max_qubits": n_qubits_threshold})
                rows.append(m)
            elif protocol == "crosssrc":
                # 跨源外推：在 MQT Bench 上训练，完全在 QCircuitBench（外部来源）上测试。
                # 不使用 algorithm 类别特征（两库算法命名不同），因此这是"纯结构特征 + 配置"的迁移测试。
                src_col = "source" if "source" in sub.columns else ("source_feat" if "source_feat" in sub.columns else None)
                if src_col is None:
                    continue
                numeric = [c for c in STRUCT_FEATURES + CONFIG_FEATURES if c in sub.columns]
                if target == "logical_qubits":
                    numeric = [c for c in numeric if c != "num_qubits"]
                tr_mask = sub[src_col] != EXTERNAL_SOURCE
                if zs:
                    tr_mask = tr_mask & ~sub["circuit_id"].isin(zs)
                te_mask = sub[src_col] == EXTERNAL_SOURCE
                if tr_mask.sum() < 50 or te_mask.sum() < 20:
                    continue
                pipe = make_model(model_name, numeric, [])
                pipe.fit(sub.loc[tr_mask, numeric], np.log1p(sub.loc[tr_mask, target]))
                pred = pipe.predict(sub.loc[te_mask, numeric])
                m = metrics(sub.loc[te_mask, target].to_numpy(dtype=float), pred)
                m.update({"target": target, "model": model_name, "protocol": "crosssrc",
                          "external_source": EXTERNAL_SOURCE})
                rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["grouped", "loso", "scale"])
    ap.add_argument("--models", nargs="+", default=["ridge", "rf", "xgb", "mlp"])
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026, 2027, 2028])
    ap.add_argument("--scale-threshold", type=int, default=16)
    ap.add_argument("--drop-clifford-train", action="store_true",
                    help="训练侧剔除逻辑 T-count 在全部 ε 下恒为 0 的全 Clifford 线路（测试集不变）")
    ap.add_argument("--tag", default="", help="结果文件后缀，用于并列保存两套配置")
    ap.add_argument("--drop-algos-train", nargs="*", default=[], help="训练侧剔除指定算法（测试集不变）")
    args = ap.parse_args()

    EVAL.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"records={len(df)} circuits={df.circuit_id.nunique()} algorithms={df.algorithm.nunique()} "
          f"qubits={df.num_qubits.min()}–{df.num_qubits.max()}", flush=True)

    zero_ids: set[str] = set()
    if args.drop_clifford_train:
        lab = pd.read_parquet(PROC / "logical_resources.parquet")[["circuit_id", "logical_t_count"]]
        zero_ids = set(lab.groupby("circuit_id").logical_t_count.max().loc[lambda s: s == 0].index)
        print(f"[drop-clifford-train] 训练侧将剔除 {len(zero_ids)} 条全 Clifford 线路（测试集不变）", flush=True)
    if args.drop_algos_train:
        alg = set(args.drop_algos_train)
        zero_ids |= set(df.loc[df.algorithm.isin(alg), "circuit_id"].unique())
        print(f"[drop-algos-train] 额外剔除算法 {sorted(alg)} 对应线路（测试集不变）", flush=True)

    all_res = []
    pred_store: list[pd.DataFrame] = []
    t0 = time.time()
    # QCircuitBench 只作为跨源测试集，绝不进入 grouped / loso / scale 的训练或抽样，
    # 这样 A/B/C 三个协议仍然只在 MQT Bench 内部比较（与阶段一结果可比），也避免外部数据泄漏。
    core = df[df["source"] != EXTERNAL_SOURCE] if "source" in df.columns else df
    for protocol in args.protocols:
        print(f"[{protocol}] ...", flush=True)
        use = df if protocol == "crosssrc" else core
        res = run_protocol(use, protocol, args.targets, args.models, args.seeds,
                           args.scale_threshold, pred_store if protocol == "grouped" else None,
                           zero_ids)
        res["dataset_records"] = len(use)
        all_res.append(res)
        print(f"    {len(res)} rows  ({time.time()-t0:.0f}s)", flush=True)

    out = pd.concat(all_res, ignore_index=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out.to_csv(EVAL / f"metrics_raw{suffix}.csv", index=False)
    # 同时按协议组合归档，避免只跑某个协议时覆盖掉完整结果表（本仓库曾因此丢过一次 270 行表）
    out.to_csv(EVAL / f"metrics_raw_{'_'.join(args.protocols)}{suffix}.csv", index=False)
    if pred_store:
        pd.concat(pred_store, ignore_index=True).to_csv(EVAL / "predictions_grouped_xgb.csv", index=False)

    agg = (out.groupby(["protocol", "target", "model"])
           .agg(mae_log_mean=("mae_log", "mean"), mae_log_std=("mae_log", "std"),
                rmse_log_mean=("rmse_log", "mean"), r2_log_mean=("r2_log", "mean"),
                r2_log_std=("r2_log", "std"), mae_orig_mean=("mae_orig", "mean"),
                mdape_mean=("mdape_pct", "mean"), smape_mean=("smape_pct", "mean"),
                n_mean=("n", "mean"))
           .reset_index())
    agg.to_csv(EVAL / f"metrics_summary{suffix}.csv", index=False)
    print("\n分组协议（grouped）汇总：")
    print(agg[agg.protocol == "grouped"]
          .pivot_table(index="target", columns="model", values="r2_log_mean").round(3).to_string())
    print("\nLOSO 汇总（F_struct 特征集）：")
    print(agg[agg.protocol == "loso"]
          .pivot_table(index="target", columns="model", values="mdape_mean").round(1).to_string())
    print("\n规模外推（scale）汇总：")
    print(agg[agg.protocol == "scale"]
          .pivot_table(index="target", columns="model", values="r2_log_mean").round(3).to_string())

    if (agg.protocol == "crosssrc").any():
        print("\n跨源外推（crosssrc：MQT Bench 训练 → QCircuitBench 测试）汇总：")
        c = agg[agg.protocol == "crosssrc"]
        print(c.pivot_table(index="target", columns="model", values="r2_log_mean").round(3).to_string())
        print(c.pivot_table(index="target", columns="model", values="mdape_mean").round(1).to_string())


if __name__ == "__main__":
    main()
