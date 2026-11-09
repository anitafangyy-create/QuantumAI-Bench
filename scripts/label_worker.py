#!/usr/bin/env python3
"""分块标注 worker：被 label_parallel.py 以子进程方式调用，处理给定的 circuit_id 列表。

单独做成 worker 的原因：个别超大规模线路（ALG 层 n≥16 的 grover、shor 等）在 qiskit
Rust transpiler 里会长时间不可中断，进程内 signal 无法打断。父进程用 subprocess
timeout 硬杀子进程，从而保证整批标注一定能跑完，只损失个别超大线路。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
from qiskit import qasm2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_io import load_circuit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_bd():
    spec = importlib.util.spec_from_file_location("bd", ROOT / "scripts" / "build_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", nargs="+", type=float, default=[1e-3, 1e-6, 1e-9])
    args = ap.parse_args()

    bd = load_bd()
    ids = [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
    manifest = pd.read_parquet(ROOT / "raw_data" / "circuits_manifest.parquet")
    sub = manifest[manifest.circuit_id.isin(ids)]

    synth = bd.AngleSynthesizer()
    feats_rows, label_rows, errs = [], [], []
    for row in sub.itertuples():
        try:
            if not row.qasm_path or not Path(row.qasm_path).exists():
                raise ValueError("no_qasm_file: 既无 QASM2 也无 QPY 落盘（见 scripts/acquire_dynamic.py）")
            if row.abstraction_level == "ALG" and int(row.requested_size) > bd.ALG_LABEL_MAX_SIZE:
                raise ValueError(f"skipped_expansion_cost: ALG n={row.requested_size}")
            qc = load_circuit(row.qasm_path)
            feats = bd.circuit_features(qc)
            structural, labels = bd.clifford_t_decompose(qc, args.eps, synth)
            feats.update(structural)
            feats.update({k: getattr(row, k) for k in
                          ["circuit_id", "source", "algorithm", "abstraction_level", "requested_size", "num_clbits"]})
            feats_rows.append(feats)
            for eps, lab in labels.items():
                label_rows.append({
                    "circuit_id": row.circuit_id, "algorithm": row.algorithm, "source": row.source,
                    "abstraction_level": row.abstraction_level, "num_qubits": feats["num_qubits"],
                    "synthesis_epsilon": eps, "logical_t_count": lab["t_count"],
                    "logical_t_depth": lab["t_depth_serial"],
                    "t_depth_structural": feats["t_depth_structural"],
                    "t_per_rotation": lab["t_per_rotation"], "logical_qubits": feats["num_qubits"],
                    "clifford_count": feats["clifford_ops_after_ct"], "cx_count": feats["cx_after_ct"],
                })
        except Exception as exc:
            errs.append({"circuit_id": row.circuit_id, "error": f"{type(exc).__name__}: {exc}"[:200]})

    out = Path(args.out)
    pd.DataFrame(feats_rows).to_parquet(out.with_suffix(".feat.parquet"), index=False)
    pd.DataFrame(label_rows).to_parquet(out.with_suffix(".label.parquet"), index=False)
    out.with_suffix(".err.json").write_text(json.dumps(errs))
    out.with_suffix(".done").write_text("ok")


if __name__ == "__main__":
    main()
