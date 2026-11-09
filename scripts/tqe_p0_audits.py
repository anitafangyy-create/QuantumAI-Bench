#!/usr/bin/env python3
"""P0 audits for the TQE revision.

Produces, into results/evaluation/:
  feature_ablation.csv   raw / basis / full feature sets, grouped + crosssrc, rf + gbdt
  zero_handling.csv      zero-T-count policy audit: counts per split, zero-class precision/recall
  shift_audit.csv        per-feature two-sample KS between MQT Bench (train) and QCircuitBench (test)
  config_grid.json       the exact Cartesian grid behind the 27 physical records per circuit
  stats_friedman_enriched.csv  Friedman output plus df, Kendall's W and Holm-adjusted p

Nothing here re-labels or re-trains the paper's headline configuration; the headline numbers stay as
published in metrics_raw_v8.csv.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import train_models as tm

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "evaluation"
PROC = ROOT / "processed_data"

POST_DECOMP = [
    "n_clifford_rz", "n_nonclifford_rz", "t_count_clifford_part",
    "t_depth_structural", "clifford_ops_after_ct", "cx_after_ct",
]
RAW_COUNTS = ["t_count_orig", "tdg_count_orig", "ccx_count_orig", "rz_count_orig"]


def feature_sets(df):
    full = [c for c in tm.STRUCT_FEATURES if c in df.columns]
    raw = [c for c in full if c not in POST_DECOMP and c not in RAW_COUNTS]
    basis = [c for c in full if c not in POST_DECOMP]
    return {"raw": raw, "basis": basis, "full": full}


def eval_split(tr, te, target, model_name, cols, tag):
    pipe = tm.make_model(model_name, cols + tm.CONFIG_FEATURES, [])
    t0 = time.time()
    pipe.fit(tr[cols + tm.CONFIG_FEATURES], np.log1p(tr[target]))
    fit_s = time.time() - t0
    t1 = time.time()
    pred = pipe.predict(te[cols + tm.CONFIG_FEATURES])
    pred_s = time.time() - t1
    m = tm.metrics(te[target].to_numpy(dtype=float), pred)
    y_true = te[target].to_numpy(dtype=float)
    y_pred = np.clip(np.expm1(np.clip(pred, -50, 20)), 0, None)
    zero = y_true == 0
    m.update({
        "featureset": tag, "n_features": len(cols),
        "fit_seconds": fit_s, "predict_ms_per_1k": pred_s * 1e3 / max(len(te), 1) * 1e3,
        "n_zero_ref": int(zero.sum()),
    })
    if zero.sum():
        zp = y_pred[zero] < 0.5
        tp = int(zp.sum())
        m["zero_recall"] = tp / int(zero.sum())
        m["zero_precision"] = tp / max(int((y_pred < 0.5).sum()), 1)
        m["zero_median_abs_log1p_err"] = float(np.median(np.abs(np.log1p(y_pred[zero]) - 0.0)))
    return m


def run_ablation(df):
    sets = feature_sets(df)
    rows = []
    for protocol in ["grouped", "crosssrc"]:
        for target in ["logical_t_count", "physical_qubits"]:
            sub = df.dropna(subset=[target]).copy()
            src = "source" if "source" in sub.columns else "source_feat"
            for tag, cols in sets.items():
                for model_name in ["rf", "xgb"]:
                    if protocol == "grouped":
                        seed = 42
                        gss = tm.GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
                        a, b = next(gss.split(sub, groups=sub["circuit_id"]))
                        tr, te = sub.iloc[a], sub.iloc[b]
                    else:
                        tr = sub[sub[src] != tm.EXTERNAL_SOURCE]
                        te = sub[sub[src] == tm.EXTERNAL_SOURCE]
                    m = eval_split(tr, te, target, model_name, cols, tag)
                    m.update({"protocol": protocol, "target": target, "model": model_name,
                              "n_train": len(tr), "n_test": len(te)})
                    rows.append(m)
                    print(f"  {protocol}/{target}/{tag}/{model_name}: "
                          f"mdape={m['mdape_pct']:.2f} r2={m['r2_log']:.3f} fit={m['fit_seconds']:.1f}s",
                          flush=True)
    return pd.DataFrame(rows)


def run_zero_audit(df):
    """Zero-target policy audit on the paper's headline split (grouped, seed 42)."""
    tgt = "logical_t_count"
    sub = df.dropna(subset=[tgt]).copy()
    src = "source" if "source" in sub.columns else "source_feat"
    gss = tm.GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    a, b = next(gss.split(sub, groups=sub["circuit_id"]))
    tr, te = sub.iloc[a], sub.iloc[b]
    cols = feature_sets(df)["full"]

    rows = []
    for model_name in ["ridge", "rf", "xgb"]:
        pipe = tm.make_model(model_name, cols + tm.CONFIG_FEATURES, [])
        pipe.fit(tr[cols + tm.CONFIG_FEATURES], np.log1p(tr[tgt]))
        pred = np.clip(np.expm1(np.clip(pipe.predict(te[cols + tm.CONFIG_FEATURES]), -50, 20)), 0, None)
        y = te[tgt].to_numpy(dtype=float)
        zero = y == 0
        zp = pred < 0.5
        tp = int((zero & zp).sum())
        rows.append({
            "model": model_name, "protocol": "grouped(seed 42)",
            "n_train_rows": len(tr), "n_train_zero": int((tr[tgt] == 0).sum()),
            "n_test_rows": len(te), "n_test_zero": int(zero.sum()),
            "zero_share_test_pct": 100 * float(zero.mean()),
            "zero_recall": tp / max(int(zero.sum()), 1),
            "zero_precision": tp / max(int(zp.sum()), 1),
            "zero_f1": 2 * tp / max(int(zero.sum()) + int(zp.sum()), 1),
            "median_abs_log1p_err_on_zero": float(np.median(np.abs(np.log1p(pred[zero])))),
            "frac_zero_pred_gt_1": float((pred[zero] > 1).mean()),
            "nonzero_mdape": float(np.median(np.minimum(
                np.abs(np.log10(np.maximum(pred[~zero] / y[~zero], 1e-300))) * 100, 1e6))),
        })
    # cross-source: does the external corpus contain any zero reference at all?
    ext = df[df[src] == tm.EXTERNAL_SOURCE]
    src2 = df[df[src] != tm.EXTERNAL_SOURCE]
    rows.append({
        "model": "n/a", "protocol": "source census",
        "n_train_rows": int((src2[tgt] == 0).sum()), "n_train_zero": int((src2[tgt] == 0).sum()),
        "n_test_rows": int((ext[tgt] == 0).sum()), "n_test_zero": int((ext[tgt] == 0).sum()),
        "zero_share_test_pct": 100 * float((ext[tgt] == 0).mean()),
    })
    return pd.DataFrame(rows)


def run_shift_audit(df):
    src = "source" if "source" in df.columns else "source_feat"
    a = df[df[src] != tm.EXTERNAL_SOURCE]
    b = df[df[src] == tm.EXTERNAL_SOURCE]
    feats = [c for c in tm.STRUCT_FEATURES if c in df.columns]
    rows = []
    for f in feats:
        x, y = a[f].dropna().to_numpy(), b[f].dropna().to_numpy()
        if len(x) < 10 or len(y) < 10:
            continue
        ks = sps.ks_2samp(x, y)
        rows.append({
            "feature": f, "mqt_median": float(np.median(x)), "qcb_median": float(np.median(y)),
            "median_ratio": float((np.median(y) + 1e-12) / (np.median(x) + 1e-12)),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "mqt_p90": float(np.percentile(x, 90)), "qcb_p90": float(np.percentile(y, 90)),
        })
    trows = []
    for t in tm.TARGETS:
        x, y = a[t].dropna().to_numpy(), b[t].dropna().to_numpy()
        trows.append({"target": t, "mqt_median": float(np.median(x)),
                      "qcb_median": float(np.median(y)),
                      "median_ratio": float((np.median(y) + 1e-12) / (np.median(x) + 1e-12)),
                      "mqt_p90": float(np.percentile(x, 90)), "qcb_p90": float(np.percentile(y, 90))})
    overlap = {
        "mqt_algorithms": int(a["algorithm"].nunique()),
        "qcb_algorithms": int(b["algorithm"].nunique()),
        "shared_algorithm_labels": sorted(set(a["algorithm"]) & set(b["algorithm"])),
        "n_shared": len(set(a["algorithm"]) & set(b["algorithm"])),
    }
    # near-duplicate audit: identical (qubits, gate_count, depth, cx) signature across sources
    cols = [c for c in ["num_qubits", "gate_count", "circuit_depth", "cx_count"] if c in df.columns]
    sig_a = set(map(tuple, a[cols].astype("int64").drop_duplicates().to_numpy()))
    sig_b = set(map(tuple, b[cols].astype("int64").drop_duplicates().to_numpy()))
    overlap["n_qcb_signatures"] = len(sig_b)
    overlap["n_qcb_signatures_matching_mqt"] = len(sig_a & sig_b)
    return pd.DataFrame(rows), pd.DataFrame(trows), overlap


def run_config_grid():
    ft = pd.read_parquet(PROC / "fault_tolerant_resources.parquet")
    grid = (ft[["synthesis_epsilon", "physical_error_rate", "logical_error_budget"]]
            .drop_duplicates().sort_values(["synthesis_epsilon", "physical_error_rate", "logical_error_budget"]))
    return {
        "n_combinations": int(len(grid)),
        "synthesis_epsilon": sorted(ft["synthesis_epsilon"].unique().tolist()),
        "physical_error_rate": sorted(ft["physical_error_rate"].unique().tolist()),
        "logical_error_budget": sorted(ft["logical_error_budget"].unique().tolist()),
        "records_per_circuit": float(len(ft) / ft["circuit_id"].nunique()),
        "nine_config_figure_subset": "the 3x3 grid of (p, budget) at epsilon = 1e-9, i.e. one ninth of the 27",
    }


def run_stats_enrich():
    fr = pd.read_csv(EVAL / "stats_friedman.csv")
    fr["df"] = 2  # k-1 models
    fr["kendall_w"] = fr["friedman_stat"] / (fr["n_blocks"] * 2)
    fr.to_csv(EVAL / "stats_friedman_enriched.csv", index=False)
    return fr


def main():
    df = tm.load_data()
    print(f"rows={len(df)} circuits={df.circuit_id.nunique()}", flush=True)

    cfg = run_config_grid()
    (EVAL / "config_grid.json").write_text(json.dumps(cfg, indent=2))
    print("config grid:", cfg["n_combinations"], flush=True)

    ab = run_ablation(df)
    ab.to_csv(EVAL / "feature_ablation.csv", index=False)

    ze = run_zero_audit(df)
    ze.to_csv(EVAL / "zero_handling.csv", index=False)

    sh, tg, ov = run_shift_audit(df)
    sh.to_csv(EVAL / "shift_audit.csv", index=False)
    tg.to_csv(EVAL / "shift_targets.csv", index=False)
    (EVAL / "shift_overlap.json").write_text(json.dumps(ov, indent=2))

    run_stats_enrich()
    print("done", flush=True)


if __name__ == "__main__":
    main()
