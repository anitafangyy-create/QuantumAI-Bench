"""按新的随机 rep 补采 `randomcircuit` 实例（同族不同实例，提升训练侧多样性）。

背景：现有 39 条 randomcircuit 全部是 `rep=0`（采集时 `--random-reps 1`），同族只有一个实例，
留一算法与跨源迁移里这一族几乎等同于一个点。补充同 n、不同 seed/rep 的新实例，成本极低。

`circuit_id` 形如 `mqt-{level}-randomcircuit-n{n}-r{rep}-{sha[:10]}`，rep 不同即新 id、天然不冲突；
`np.random.seed(seed+rep)` 与 `build_dataset.py` 完全一致，保证可复现。

用法：
    .venv/bin/python scripts/acquire_randomcircuit.py --reps 1 20
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import qasm2, qpy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import gate_statistics, sha256_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "raw_data" / "circuits_manifest.parquet"
QASM_DIR = ROOT / "raw_data" / "circuits"

INDEP_SIZES = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24]
ALG_SIZES = [6, 8, 10, 12, 14, 16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="必须与 build_dataset 采集时一致")
    ap.add_argument("--reps", nargs=2, type=int, default=[1, 20], metavar=("FROM", "TO"))
    ap.add_argument("--max-size", type=int, default=24)
    args = ap.parse_args()

    from mqt.bench import get_benchmark, BenchmarkLevel

    level_map = {"ALG": BenchmarkLevel.ALG, "INDEP": BenchmarkLevel.INDEP}
    manifest = pd.read_parquet(MANIFEST)
    have = set(manifest.circuit_id)
    QASM_DIR.mkdir(parents=True, exist_ok=True)

    new_rows, failed = [], []
    for level_name, sizes in (("INDEP", INDEP_SIZES), ("ALG", ALG_SIZES)):
        for n in [s for s in sizes if s <= args.max_size]:
            for rep in range(args.reps[0], args.reps[1] + 1):
                try:
                    np.random.seed(args.seed + rep)
                    random.seed(args.seed + rep)
                    qc = get_benchmark("randomcircuit", level_map[level_name], n,
                                       opt_level=0, random_parameters=True)
                    stats = gate_statistics(qc)
                    if qc.num_qubits == 0 or stats["gate_count"] == 0:
                        continue
                    try:
                        qasm_str = qasm2.dumps(qc)
                        status, fmt = "ok", "qasm2"
                    except Exception:
                        qasm_str, status, fmt = "", "qasm2_dump_failed", "qpy"
                    digest = sha256_text(qasm_str) if qasm_str else sha256_text(repr(qc.data[:5]))
                    cid = f"mqt-{level_name.lower()}-randomcircuit-n{n}-r{rep}-{digest[:10]}"
                    if cid in have:
                        continue
                    path = QASM_DIR / (f"{cid}.qasm" if qasm_str else f"{cid}.qpy")
                    if qasm_str:
                        path.write_text(qasm_str)
                    else:
                        with open(path, "wb") as f:
                            qpy.dump(qc, f)
                    new_rows.append({
                        "circuit_id": cid, "source": "MQT_Bench", "algorithm": "randomcircuit",
                        "abstraction_level": level_name, "requested_size": n, "rep": rep,
                        "num_qubits": qc.num_qubits, "num_clbits": qc.num_clbits,
                        "qasm_sha256": digest, "qasm_path": str(path),
                        "qasm_status": status, "qasm_format": fmt,
                        "generation_seconds": 0.0, "status": "ok",
                    })
                    have.add(cid)
                except Exception as exc:  # noqa: BLE001
                    failed.append({"level": level_name, "requested_size": n, "rep": rep,
                                   "error": f"{type(exc).__name__}: {exc}"[:200]})

    if new_rows:
        add = pd.DataFrame(new_rows)
        for col in ("qasm_format",):
            if col not in manifest.columns:
                manifest[col] = None
        manifest = pd.concat([manifest, add.reindex(columns=manifest.columns)], ignore_index=True)
        backup = MANIFEST.with_suffix(".backup_before_randomcircuit.parquet")
        if not backup.exists():
            pd.read_parquet(MANIFEST).to_parquet(backup, index=False)
        manifest.to_parquet(MANIFEST, index=False)

    print(f"[done] 新增 {len(new_rows)} 条 / 失败 {len(failed)} / manifest {len(manifest)}", flush=True)
    if failed:
        pd.DataFrame(failed).to_csv(ROOT / "logs" / "acquire_randomcircuit_failed.csv", index=False)


if __name__ == "__main__":
    main()
