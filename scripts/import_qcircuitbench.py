#!/usr/bin/env python3
"""把 QCircuitBench（OpenQASM 3.0）接入 QuantumAI-FTQC 流水线。

为什么要单独一个脚本：
  - QCircuitBench 是 OpenQASM 3.0，且大量文件 include 非 stdgates 的自定义头文件；
    qiskit.qasm3 直接 loads 会抛 "non-stdgates imports not currently supported"。
  - 现有流水线（label_parallel.py）只认 OpenQASM2 + manifest，所以这里负责把 qasm3
    解析出来、展开到 {rz, sx, x, cx} 基、再落成 OpenQASM2 文件并追加 manifest 行。
  - 转换本身要在子进程里做：超大线路（n>100）的 qiskit transpile 在进程内无法被
    SIGALRM 打断，父进程用 subprocess 超时硬杀，损失可量化。

用法：
  # 试点（200 条、≤40 比特）
  .venv/bin/python scripts/import_qcircuitbench.py --limit 200 --max-qubits 40

  # 全量
  .venv/bin/python scripts/import_qcircuitbench.py --max-qubits 60

产出：
  raw_data/circuits/qcb_*.qasm                 OpenQASM2 线路
  raw_data/circuits_manifest.parquet           追加 source=qcircuitbench 的行
  logs/qcb_import_*.json / *_failures.csv      转换结果与失败明细

之后直接跑：
  .venv/bin/python scripts/label_parallel.py --chunk 5 --timeout 120 --workers 4
就会被自动接管（它按 manifest 里未标注的 circuit_id 派活）。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QCB = ROOT / "raw_data" / "QCircuitBench"
QASM_OUT = ROOT / "raw_data" / "circuits"
LOG = ROOT / "logs"
MANIFEST = ROOT / "raw_data" / "circuits_manifest.parquet"

SKIP_DIR = re.compile(r"^(n\d+|trial\d+|\d+|test_oracle|.*_oracle|oracle)$", re.I)


def algorithm_from_path(path: str) -> str:
    """从目录结构推断算法名：跳过 n12 / trial3 / test_oracle 这类层级。"""
    rel = os.path.relpath(path, QCB)
    parts = [p for p in Path(rel).parts[:-1]]          # 去掉文件名
    cats = {"Algorithm_Design", "benchmark_subset", "Quantum_Computing",
            "Quantum_Information", "Variational_Circuits", "qasm_oracle_construction",
            "universal", "clifford"}
    named = [p for p in parts if p not in cats and not SKIP_DIR.match(p)]
    if named:
        return named[-1]
    return parts[-1] if parts else "unknown"


def inline_qasm(path: str, depth: int = 0) -> str:
    """递归内联 include：非 stdgates 的头文件按同目录 → 全库 basename 查找。"""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    if depth > 2:
        return txt

    def rep(m: re.Match[str]) -> str:
        name = m.group(1)
        if "stdgates" in name:
            return m.group(0)
        cand = os.path.join(os.path.dirname(path), name)
        if not os.path.exists(cand):
            hits = glob.glob(f"{QCB}/**/{os.path.basename(name)}", recursive=True)
            cand = hits[0] if hits else None
        if not cand:
            return ""
        return inline_qasm(cand, depth + 1)

    txt = re.sub(r'include\s+"([^"]+)";', rep, txt)
    if "stdgates.inc" not in txt:
        txt = txt.replace("OPENQASM 3.0;", 'OPENQASM 3.0;\ninclude "stdgates.inc";', 1)
        txt = txt.replace("OPENQASM 3;", 'OPENQASM 3;\ninclude "stdgates.inc";', 1)
    return txt


QUBIT_PAT = re.compile(r"qubit\[(\d+)\]")


def cheap_num_qubits(path: str) -> int | None:
    m = QUBIT_PAT.search(Path(path).read_text(encoding="utf-8", errors="ignore"))
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------------- worker
def run_worker(ids_file: str) -> None:
    from qiskit import qasm2, qasm3, transpile

    ids = [line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()]
    out = []
    for path in ids:
        rec = {"source_path": path, "algorithm": algorithm_from_path(path)}
        try:
            qc = qasm3.loads(inline_qasm(path))
            # 统一到 {rz, sx, x, cx}：qasm2.dumps 对该基的所有门都有定义，后续也能直接复算
            base = transpile(qc, basis_gates=["rz", "sx", "x", "cx"],
                             optimization_level=0, seed_transpiler=0)
            text = qasm2.dumps(base)
            digest = hashlib.sha256(text.encode()).hexdigest()
            circuit_id = f"qcb_{digest[:12]}"
            out_path = QASM_OUT / f"{circuit_id}.qasm"
            if not out_path.exists():
                out_path.write_text(text)
            rec.update({
                "status": "ok",
                "circuit_id": circuit_id,
                "qasm_sha256": digest,
                "qasm_path": str(out_path),
                "num_qubits": base.num_qubits,
                "num_clbits": base.num_clbits,
                "gate_count": sum(base.count_ops().values()),
            })
        except Exception as exc:
            rec.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}"[:180]})
        out.append(rec)
    Path(ids_file).with_suffix(".out.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"[worker] done {len(out)}", flush=True)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条（按路径排序，确定性抽样）")
    ap.add_argument("--max-qubits", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=90, help="单个转换分块的墙钟上限（秒）")
    ap.add_argument("--seed", type=int, default=2026, help="抽样随机种子（--sample 时生效）")
    ap.add_argument("--sample", action="store_true", help="随机抽样而非取前 N 条")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行转换的分块数（每个分块一个子进程）。本机 8 核用 1–2，128 核可给 24–32")
    ap.add_argument("--skip-processed", action="store_true",
                    help="跳过 logs/qcb_processed_paths.txt 里已尝试过的源文件（断点续跑，避免重复转换）")
    # 内部：子进程模式
    ap.add_argument("--worker-ids", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.worker_ids:
        run_worker(args.worker_ids)
        return

    QASM_OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")

    files = sorted(glob.glob(f"{QCB}/**/*.qasm", recursive=True))
    print(f"QCircuitBench qasm 总数 {len(files)}", flush=True)

    # 先用正则读比特数，把超大规模线路挡在解析之前（解析本身不贵，transpile 才是坑）
    keep, skipped_big = [], 0
    for f in files:
        n = cheap_num_qubits(f)
        if n is None or n > args.max_qubits:
            skipped_big += 1
            continue
        keep.append(f)
    print(f"≤{args.max_qubits} 比特候选 {len(keep)}（跳过 {skipped_big}）", flush=True)

    if args.skip_processed:
        done_file = LOG / "qcb_processed_paths.txt"
        if done_file.exists():
            done = set(done_file.read_text().splitlines())
            before = len(keep)
            keep = [f for f in keep if f not in done]
            print(f"跳过已尝试过的源文件 {before - len(keep)} 条 → 剩余候选 {len(keep)}", flush=True)

    if args.sample:
        import random
        random.seed(args.seed)
        random.shuffle(keep)
    if args.limit:
        keep = keep[: args.limit]

    chunks = [keep[i:i + args.chunk] for i in range(0, len(keep), args.chunk)]
    print(f"待转换 {len(keep)} 条 → {len(chunks)} 个分块（chunk={args.chunk}, timeout={args.timeout}s, "
          f"workers={args.workers}）", flush=True)

    records, failures = [], []
    t0 = time.time()
    done_paths: list[str] = []
    lock = threading.Lock()

    def run_chunk(k: int, ch: list[str]) -> None:
        ids_file = LOG / f"qcb_chunk_{tag}_{k}.txt"
        ids_file.write_text("\n".join(ch))
        recs, fails = [], []
        try:
            subprocess.run([sys.executable, __file__, "--worker-ids", str(ids_file)],
                           timeout=args.timeout, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            res = json.loads(ids_file.with_suffix(".out.json").read_text())
            recs = [r for r in res if r["status"] == "ok"]
            fails = [r for r in res if r["status"] == "fail"]
        except subprocess.TimeoutExpired:
            fails = [{"source_path": p, "status": "fail",
                      "error": f"chunk_timeout: >{args.timeout}s"} for p in ch]
        except Exception as exc:
            fails = [{"source_path": p, "status": "fail",
                      "error": f"chunk_error: {exc}"[:180]} for p in ch]
        with lock:
            records.extend(recs)
            failures.extend(fails)
            done_paths.extend(ch)
            n = len(records) + len(failures)
            if k % 5 == 0 or k == len(chunks):
                print(f"  chunk {k}/{len(chunks)} 完成 {n}/{len(keep)} "
                      f"(ok={len(records)} fail={len(failures)}, {time.time()-t0:.0f}s)", flush=True)

    if args.workers <= 1:
        for k, ch in enumerate(chunks, 1):
            run_chunk(k, ch)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_chunk, k, ch) for k, ch in enumerate(chunks, 1)]
            for f in as_completed(futs):
                f.result()

    # 记录已尝试过的源文件，供 --skip-processed 断点续跑
    if done_paths:
        p = LOG / "qcb_processed_paths.txt"
        with p.open("a") as fh:
            fh.write("\n".join(done_paths) + "\n")

    df = pd.DataFrame(records)
    if len(df):
        df["source"] = "qcircuitbench"
        df["abstraction_level"] = "EXT"
        df["requested_size"] = df.num_qubits
        df["rep"] = 0
        df["qasm_status"] = "ok"
        df["generation_seconds"] = 0.0
        df["status"] = "ok"
        cols = ["circuit_id", "source", "algorithm", "abstraction_level", "requested_size",
                "rep", "num_qubits", "num_clbits", "qasm_sha256", "qasm_path", "qasm_status",
                "generation_seconds", "status"]
        new = df[cols].drop_duplicates(subset=["qasm_sha256"])

        if MANIFEST.exists():
            base = pd.read_parquet(MANIFEST)
            merged = pd.concat([base, new], ignore_index=True).drop_duplicates(subset=["qasm_sha256"])
            added = len(merged) - len(base)
        else:
            merged, added = new, len(new)
        merged.to_parquet(MANIFEST, index=False)
        print(f"manifest: {len(base) if MANIFEST.exists() else 0} → {len(merged)} 行（新增 {added}，按 sha256 去重）", flush=True)

    pd.DataFrame(failures).to_csv(LOG / f"qcb_import_failures_{tag}.csv", index=False)
    summary = {
        "tag": tag, "max_qubits": args.max_qubits, "limit": args.limit,
        "candidates": len(keep), "ok": len(records), "fail": len(failures),
        "skipped_big": skipped_big, "elapsed_s": round(time.time() - t0, 1),
        "fail_reasons": pd.Series([f.get("error", "")[:40] for f in failures]).value_counts().head(5).to_dict()
              if failures else {},
    }
    (LOG / f"qcb_import_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
