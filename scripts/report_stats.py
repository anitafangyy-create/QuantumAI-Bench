#!/usr/bin/env python3
"""统计数据集规模并落盘 JSON：DATASET_CARD / 阶段报告里的每个数字都由本脚本复算。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
RAW = ROOT / "raw_data"
EVAL = ROOT / "results" / "evaluation"


def main() -> None:
    man = pd.read_parquet(RAW / "circuits_manifest.parquet")
    feat = pd.read_parquet(PROC / "circuit_features.parquet")
    lab = pd.read_parquet(PROC / "logical_resources.parquet")
    ft = pd.read_parquet(PROC / "fault_tolerant_resources.parquet")

    stats = {
        "manifest_rows": int(len(man)),
        "manifest_unique_circuit_ids": int(man.circuit_id.nunique()),
        "manifest_duplicate_hashes": int(man.qasm_sha256.duplicated().sum()),
        "labeled_circuits": int(feat.circuit_id.nunique()),
        "unlabeled_circuits": int(man.circuit_id.nunique() - feat.circuit_id.nunique()),
        "logical_label_rows": int(len(lab)),
        "ft_label_rows": int(len(ft)),
        "unique_algorithms": int(feat.algorithm.nunique()),
        "algorithms": sorted(feat.algorithm.unique().tolist()),
        "abstraction_levels": {k: int(v) for k, v in feat.abstraction_level.value_counts().items()},
        "num_qubits_min": int(feat.num_qubits.min()),
        "num_qubits_max": int(feat.num_qubits.max()),
        "synthesis_epsilon": sorted(float(x) for x in lab.synthesis_epsilon.unique()),
        "physical_error_rate": sorted(float(x) for x in ft.physical_error_rate.unique()),
        "logical_error_budget": sorted(float(x) for x in ft.logical_error_budget.unique()),
        "unique_configs": int(ft.groupby(["synthesis_epsilon", "physical_error_rate", "logical_error_budget"]).ngroups),
        "t_count_min": int(lab.logical_t_count.min()),
        "t_count_max": int(lab.logical_t_count.max()),
        "t_count_zero_circuits": int((lab[lab.synthesis_epsilon == lab.synthesis_epsilon.min()].logical_t_count == 0).sum()),
        "code_distance_min": int(ft.code_distance.min()),
        "code_distance_max": int(ft.code_distance.max()),
        "physical_qubits_max": float(ft.physical_qubits.max()),
        "runtime_seconds_max": float(ft.runtime_seconds.max()),
        "ft_status_counts": {k: int(v) for k, v in ft.status.value_counts().items()},
        "qasm_files_on_disk": int(len(list((RAW / "circuits").glob("*.qasm")))) if (RAW / "circuits").exists() else 0,
    }

    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "algorithms"}, indent=2, ensure_ascii=False))
    print("\n复现口径：manifest / processed_data 三层 parquet 直接统计，不含任何推算。")


if __name__ == "__main__":
    main()
