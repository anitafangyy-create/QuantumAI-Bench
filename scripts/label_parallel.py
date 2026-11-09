#!/usr/bin/env python3
"""分块并行标注驱动：把待标注线路切成小块，每块一个子进程，超时硬杀并继续。

这样即使个别超大线路让 qiskit transpiler 卡住，整批也能跑完，损失可量化（记在
logs/label_timeouts.csv）。用法：
  .venv/bin/python scripts/label_parallel.py --chunk 10 --timeout 240 --workers 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
LOG = ROOT / "logs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=240, help="单个分块的墙钟上限（秒）")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--source", default=None, help="只标注指定 source（如 qcircuitbench），留空为全部")
    ap.add_argument("--max-qubits", type=int, default=None, help="跳过比特数超过该值的线路（按 qasm 头判断）")
    ap.add_argument("--ids-file", default=None, help="只标注该文件里列出的 circuit_id（每行一个）")
    args = ap.parse_args()

    manifest = pd.read_parquet(ROOT / "raw_data" / "circuits_manifest.parquet")
    feat_path, lab_path = PROC / "circuit_features.parquet", PROC / "logical_resources.parquet"
    feats = pd.read_parquet(feat_path) if feat_path.exists() else pd.DataFrame()
    labels = pd.read_parquet(lab_path) if lab_path.exists() else pd.DataFrame()
    done = set(feats.circuit_id) if len(feats) else set()
    pool = manifest[~manifest.circuit_id.isin(done)].copy()
    if args.source:
        pool = pool[pool.source == args.source]
    if args.ids_file:
        ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()]
        pool = pool[pool.circuit_id.isin(ids)]
        print(f"ids-file 指定 {len(ids)} 条，池内命中 {len(pool)} 条", flush=True)
    if args.max_qubits:
        import re

        def _nq(path):
            if not isinstance(path, str) or not path:
                return None
            if path.lower().endswith(".qpy"):
                try:
                    from qiskit import qpy
                    with open(path, "rb") as f:
                        qc = qpy.load(f)
                    return (qc[0] if isinstance(qc, list) else qc).num_qubits
                except Exception:
                    return None
            try:
                head = open(path).read(400)
            except (OSError, UnicodeDecodeError):
                return None
            mm = re.search(r"(?:qreg\s+\w+|qubit)\s*\[(\d+)\]", head)
            return int(mm.group(1)) if mm else None

        pool["_nq"] = pool.qasm_path.map(_nq)
        small = pool[pool._nq.isna() | (pool._nq <= args.max_qubits)]
        skipped = pool[pool._nq > args.max_qubits]
        if len(skipped):
            skipped[["circuit_id", "qasm_path", "_nq"]].to_csv(LOG / "label_skipped_oversize.csv", index=False)
            print(f"跳过 {len(skipped)} 条超限线路（>{args.max_qubits} 比特），已记 logs/label_skipped_oversize.csv", flush=True)
        pool = small
    pending = list(pool.circuit_id)
    print(f"pending {len(pending)} circuits | chunk={args.chunk} timeout={args.timeout}s "
          f"source={args.source or 'all'}", flush=True)
    if not pending:
        return

    tmpdir = Path(tempfile.mkdtemp(prefix="labelchunks_"))
    chunks = [pending[i:i + args.chunk] for i in range(0, len(pending), args.chunk)]
    running: list[list] = []          # [proc, marker, t0, ids]
    problems: list[dict] = []
    t_start = time.time()
    idx = 0
    last_flush = time.time()

    def merge(marker: Path) -> None:
        nonlocal feats, labels
        f_p = marker.with_suffix(".feat.parquet")
        if f_p.exists() and f_p.stat().st_size > 0:
            f = pd.read_parquet(f_p)
            if len(f):
                feats = pd.concat([feats, f], ignore_index=True).drop_duplicates(subset=["circuit_id"])
        l_p = marker.with_suffix(".label.parquet")
        if l_p.exists() and l_p.stat().st_size > 0:
            l = pd.read_parquet(l_p)
            if len(l):
                labels = pd.concat([labels, l], ignore_index=True)
        e_p = marker.with_suffix(".err.json")
        if e_p.exists():
            try:
                errs = json.loads(e_p.read_text() or "[]")
                problems.extend(errs)
            except json.JSONDecodeError:
                pass

    def harvest() -> None:
        for entry in list(running):
            proc, marker, t0, ids = entry
            rc = proc.poll()
            if rc is None:
                if time.time() - t0 > args.timeout:
                    proc.kill(); proc.wait()
                    problems.append({"circuit_ids": ";".join(ids),
                                     "error": f"chunk_timeout: >{args.timeout}s (qiskit 不可中断)"})
                    print(f"  [timeout] dropped chunk {ids[0]} … ({len(ids)} circuits)", flush=True)
                    running.remove(entry)
                continue
            merge(marker)
            running.remove(entry)

    while idx < len(chunks) or running:
        while idx < len(chunks) and len(running) < args.workers:
            ids = chunks[idx]
            ids_file = tmpdir / f"ids_{idx}.txt"
            ids_file.write_text("\n".join(ids))
            marker = tmpdir / f"chunk_{idx}"
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "label_worker.py"),
                 "--ids-file", str(ids_file), "--out", str(marker)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running.append([proc, marker, time.time(), ids])
            idx += 1
        harvest()
        if time.time() - last_flush > 30 and len(feats):
            feats.to_parquet(feat_path, index=False)
            if len(labels):
                labels.to_parquet(lab_path, index=False)
            last_flush = time.time()
            print(f"  … {len(feats)} circuits labelled, {len(problems)} problems", flush=True)
        time.sleep(0.3)

    feats.to_parquet(feat_path, index=False)
    labels.to_parquet(lab_path, index=False)
    pd.DataFrame(problems).to_csv(LOG / "label_timeouts.csv", index=False)
    print(f"[done] circuits={len(feats)} label_rows={len(labels)} "
          f"problems={len(problems)} elapsed={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
