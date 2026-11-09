#!/usr/bin/env python3
"""种子方差统计：哪些协议的指标真的带种子随机性，哪些是确定性划分。

背景：协议 A（grouped）跑 5 个种子；B/C/D（loso/scale/crosssrc）是确定性划分，没有种子维度。
因此"种子方差"只在 grouped 上成立，本脚本把这一点量化并写进 JSON，避免报告里含糊地说"5 seeds 均值"。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "evaluation"


def main() -> None:
    m = pd.read_csv(EVAL / "metrics_raw.csv")
    rows = []
    for protocol, g in m.groupby("protocol"):
        n_seeds = int(g["seed"].nunique()) if "seed" in g else 0
        for (target, model), gg in g.groupby(["target", "model"]):
            # loso 每个算法一行（无 seed 维度），它的离散度是"算法间"，不能当种子方差
            has_seed_repeats = n_seeds > 1 and len(gg) > 1
            rows.append({
                "protocol": protocol, "target": target, "model": model,
                "n_seeds": n_seeds,
                "r2_log_mean": float(gg["r2_log"].mean()),
                "r2_log_std": float(gg["r2_log"].std(ddof=1)) if has_seed_repeats else 0.0,
                "mdape_mean": float(gg["mdape_pct"].mean()),
                "mdape_std": float(gg["mdape_pct"].std(ddof=1)) if has_seed_repeats else 0.0,
                "mdape_rel_std_pct": float(gg["mdape_pct"].std(ddof=1) / max(abs(gg["mdape_pct"].mean()), 1e-9) * 100) if has_seed_repeats else 0.0,
                "n_rows": int(len(gg)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(EVAL / "seed_variance.csv", index=False)

    per_proto = (df.groupby("protocol")
                 .agg(n_seeds=("n_seeds", "max"),
                      max_mdape_rel_std_pct=("mdape_rel_std_pct", "max"),
                      median_mdape_rel_std_pct=("mdape_rel_std_pct", "median"))
                 .reset_index())
    out = {
        "note": "只有 grouped 具备 5 个种子；loso/scale/crosssrc 为确定性划分，种子方差恒为 0（不代表稳定，只代表没有重复）。",
        "protocols": per_proto.to_dict(orient="records"),
        "detail_file": "seed_variance.csv",
    }
    with open(EVAL / "seed_variance.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(per_proto.round(4).to_string(index=False))
    print("\ngrouped 各目标种子间 MdAPE 相对标准差（%，越小越稳）：")
    gsel = df[(df.protocol == "grouped")].sort_values(["target", "mdape_rel_std_pct"], ascending=[True, False])
    print(gsel[["target", "model", "mdape_mean", "mdape_std", "mdape_rel_std_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
