"""把 qiskit 无法导出 OpenQASM 2 的 MQT 基准线路补采为 QPY。

背景：`build_dataset.py` 用 `qasm2.dumps` 落盘，遇到带经典条件分支的线路
（`dynamic_qft` / `ghz_dynamic` / `ae` / `iqpe`）会抛 QASM2ExportError，这些线路
此前一律记 `qasm_status=qasm2_dump_failed`、不留文件，于是在标注阶段被跳过——
这才是"336 条 ALG 未标注"的真实原因，而不是 `ALG_LABEL_MAX_SIZE` 截断。

做法：用与 `build_dataset.py` 完全相同的随机种子（`seed + rep`）重新生成同一条线路，
核对比特数后写成 `.qpy`，并把 manifest 的 `qasm_path / qasm_status / qasm_format` 就地更新。
重生成是否"同一条"由 `num_qubits == manifest.num_qubits` 兜底校验；不一致的行不改，单独列出。

用法：
    .venv/bin/python scripts/acquire_dynamic.py --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import qpy

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "raw_data" / "circuits_manifest.parquet"
QASM_DIR = ROOT / "raw_data" / "circuits"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="必须与 build_dataset 采集时一致")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from mqt.bench import get_benchmark, BenchmarkLevel

    level_map = {"ALG": BenchmarkLevel.ALG, "INDEP": BenchmarkLevel.INDEP}
    manifest = pd.read_parquet(MANIFEST)
    todo = manifest[(manifest.source == "MQT_Bench") & (manifest.qasm_status != "ok")]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"待补采 {len(todo)} 条（qasm_status != ok）", flush=True)

    QASM_DIR.mkdir(parents=True, exist_ok=True)
    updated, mismatched, failed = [], [], []
    for row in todo.itertuples():
        algo, level, n, rep = row.algorithm, row.abstraction_level, int(row.requested_size), int(row.rep)
        try:
            np.random.seed(args.seed + rep)
            random.seed(args.seed + rep)
            qc = get_benchmark(algo, level_map[level], n, opt_level=0, random_parameters=True)
            if qc.num_qubits != int(row.num_qubits):
                mismatched.append({"circuit_id": row.circuit_id, "manifest_qubits": int(row.num_qubits),
                                   "regen_qubits": qc.num_qubits})
                continue
            path = QASM_DIR / f"{row.circuit_id}.qpy"
            with open(path, "wb") as f:
                qpy.dump(qc, f)
            updated.append({"circuit_id": row.circuit_id, "qasm_path": str(path),
                            "qasm_status": "qpy_ok", "qasm_format": "qpy"})
        except Exception as exc:  # noqa: BLE001
            failed.append({"circuit_id": row.circuit_id,
                           "error": f"{type(exc).__name__}: {exc}"[:200]})

    if updated:
        upd = pd.DataFrame(updated).set_index("circuit_id")
        backup = MANIFEST.with_suffix(".backup_before_qpy.parquet")
        if not backup.exists():
            manifest.to_parquet(backup, index=False)
        for col in ("qasm_path", "qasm_status", "qasm_format"):
            if col not in manifest.columns:
                manifest[col] = None
            manifest.loc[manifest.circuit_id.isin(upd.index), col] = \
                upd[col].reindex(manifest.loc[manifest.circuit_id.isin(upd.index), "circuit_id"]).values
        manifest.to_parquet(MANIFEST, index=False)

    print(f"[done] 补采成功 {len(updated)} / 比特数不符 {len(mismatched)} / 生成失败 {len(failed)}", flush=True)
    if mismatched:
        pd.DataFrame(mismatched).to_csv(ROOT / "logs" / "acquire_dynamic_mismatch.csv", index=False)
        print(f"  比特数不符明细 → logs/acquire_dynamic_mismatch.csv", flush=True)
    if failed:
        pd.DataFrame(failed).to_csv(ROOT / "logs" / "acquire_dynamic_failed.csv", index=False)
        print(f"  生成失败明细 → logs/acquire_dynamic_failed.csv", flush=True)


if __name__ == "__main__":
    main()
