#!/usr/bin/env python3
"""QuantumAI-FTQC 数据集构建流水线（阶段一：线路采集 + 逻辑资源标注）

数据源：
  - MQT Bench（ALG / INDEP 两个抽象层）
  - QASMBench（外部 OpenQASM 线路，用于跨源验证）

输出：
  raw_data/circuits/*.qasm                     原始线路（OpenQASM2）
  raw_data/circuits_manifest.parquet           线路清单（来源/算法/规模/hash）
  processed_data/circuit_features.parquet      结构 + 拓扑特征
  processed_data/logical_resources.parquet     Clifford+T 逻辑资源标签（按 synthesis_epsilon）

标签定义（严格区分“结构量”和“精度相关量”）：
  t_count_clifford_part : 只统计 Clifford 角（θ = kπ/4）旋转所对应的 T 门数，精确
  n_nonclifford_rz      : 非 Clifford 角旋转个数，需 Ross–Selinger 合成 → T 门数随 ε 变化
  t_count(ε)            : t_count_clifford_part + n_nonclifford_rz * T_PER_ROT(ε)
  t_depth_structural    : 每个旋转只占 1 个 T 层时的 T 层深（与 ε 无关，结构量）
  t_depth_serial        : 每个旋转占 T_PER_ROT(ε) 个连续 T 层的串行下界（与 ε 相关）
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from qiskit import QuantumCircuit, transpile, qasm2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_io import load_circuit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QASM_DIR = ROOT / "raw_data" / "circuits"
RAW_DIR = ROOT / "raw_data"
PROC_DIR = ROOT / "processed_data"
LOG_DIR = ROOT / "logs"

EXCLUDE_OPS = {"measure", "barrier", "reset", "id", "delay"}
CT_EXACT_TOL = 1e-8  # 判定 θ 是否为 kπ/4 的容差


# ----------------------------------------------------------------------------- utils
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CircuitTimeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: int):
    """给单条线路的处理加硬超时（macOS/Linux 信号实现）。"""
    def _handler(signum, frame):
        raise CircuitTimeout(f"exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def strip_nonunitary(qc: QuantumCircuit) -> QuantumCircuit:
    """去掉 measure/barrier/reset，得到纯酉线路（用于深度与门统计）。"""
    out = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for inst in qc.data:
        if inst.operation.name in {"measure", "barrier", "reset"}:
            continue
        qargs = [out.qubits[qc.find_bit(q).index] for q in inst.qubits]
        cargs = [out.clbits[qc.find_bit(c).index] for c in inst.clbits]
        out.append(inst.operation, qargs, cargs)
    return out


def gate_statistics(qc: QuantumCircuit) -> dict:
    """基于每个操作实际作用的量子比特数分类单/双/多比特门。"""
    one_q = two_q = multi_q = 0
    gate_counts: dict[str, int] = {}
    for inst in qc.data:
        name = inst.operation.name
        if name in EXCLUDE_OPS:
            if name == "measure":
                gate_counts["measure"] = gate_counts.get("measure", 0) + 1
            continue
        arity = len(inst.qubits)
        if arity == 1:
            one_q += 1
        elif arity == 2:
            two_q += 1
        else:
            multi_q += 1
        gate_counts[name] = gate_counts.get(name, 0) + 1
    return {
        "one_qubit_gates": one_q,
        "two_qubit_gates": two_q,
        "multi_qubit_gates": multi_q,
        "gate_count": one_q + two_q + multi_q,
        "gate_counts": gate_counts,
    }


def interaction_graph_stats(qc: QuantumCircuit) -> dict:
    n = qc.num_qubits
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for inst in qc.data:
        if inst.operation.name in EXCLUDE_OPS:
            continue
        if len(inst.qubits) == 2:
            q0, q1 = (qc.find_bit(q).index for q in inst.qubits)
            if g.has_edge(q0, q1):
                g[q0][q1]["weight"] = g[q0][q1].get("weight", 0) + 1
            else:
                g.add_edge(q0, q1, weight=1)
    degrees = [d for _, d in g.degree()]
    comps = list(nx.connected_components(g))
    return {
        "interaction_edges": g.number_of_edges(),
        "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
        "degree_max": int(np.max(degrees)) if degrees else 0,
        "graph_density": float(nx.density(g)) if n > 1 else 0.0,
        "connected_components": len(comps),
        "largest_component": max((len(c) for c in comps), default=0),
        "weighted_edge_sum": int(sum(d["weight"] for _, _, d in g.edges(data=True))) if g.number_of_edges() else 0,
    }


def angle_class(angle: float) -> tuple[bool, int]:
    """判断 RZ(angle) 是否落在 Clifford 角 kπ/4 上，并返回该角的精确 T 门数。

    注意：Clifford 角并非都可以用 T 门实现。θ=(kπ/4)Rz 的精确 T 代价为
      k 为偶数 → 0（k=0:I, k=2:S, k=4:Z, k=6:S†，全部 Clifford）
      k 为奇数 → 1（T 或 T† 的整数幂）
    早期版本用 min(k, 8-k) 会把 S 门（k=2）算成 2 个 T，导致全 Clifford 线路
    （如 GHZ / graphstate）T-count 非零，属于标签定义错误。
    """
    k = angle / (math.pi / 4)
    if abs(k - round(k)) < CT_EXACT_TOL * 8:
        nearest = ((round(k) % 8) + 8) % 8
        return True, (1 if nearest % 2 == 1 else 0)
    return False, 0


# ----------------------------------------------------------------------------- acquisition
MQT_SIZE_GRID = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28]
ALG_LABEL_MAX_SIZE = 20  # ALG（黑盒算子级）线路可完成标注的最大规模，见主循环注释


def acquire_mqt(levels, max_circuits: int | None, random_reps: int, seed: int,
                sizes: list[int] | None = None) -> list[dict]:
    from mqt.bench import get_benchmark, BenchmarkLevel
    from mqt.bench.benchmarks import get_available_benchmark_names

    level_map = {"ALG": BenchmarkLevel.ALG, "INDEP": BenchmarkLevel.INDEP}
    algos = get_available_benchmark_names()
    size_grid = sizes if sizes else MQT_SIZE_GRID
    records, failures = [], []
    rng = random.Random(seed)

    for level_name in levels:
        level = level_map[level_name]
        for algo in algos:
            reps = random_reps if algo == "randomcircuit" else 1
            for n in size_grid:
                for rep in range(reps):
                    if max_circuits is not None and len(records) >= max_circuits:
                        return records, failures
                    t0 = time.time()
                    try:
                        np.random.seed(seed + rep)
                        random.seed(seed + rep)
                        qc = get_benchmark(algo, level, n, opt_level=0, random_parameters=True)
                        if qc.num_qubits == 0:
                            raise ValueError("empty circuit")
                        stats = gate_statistics(qc)
                        if stats["gate_count"] == 0:
                            raise ValueError("no gates")
                        try:
                            qasm_str = qasm2.dumps(qc)
                            qasm_status = "ok"
                        except Exception:
                            qasm_str = ""
                            qasm_status = "qasm2_dump_failed"
                        digest = sha256_text(qasm_str) if qasm_str else sha256_text(repr(qc.data[:5]))
                        circuit_id = f"mqt-{level_name.lower()}-{algo}-n{n}-r{rep}-{digest[:10]}"
                        path = ""
                        if qasm_str:
                            path = str(QASM_DIR / f"{circuit_id}.qasm")
                            with open(path, "w") as f:
                                f.write(qasm_str)
                        records.append({
                            "circuit_id": circuit_id,
                            "source": "MQT_Bench",
                            "algorithm": algo,
                            "abstraction_level": level_name,
                            "requested_size": n,
                            "rep": rep,
                            "num_qubits": qc.num_qubits,
                            "num_clbits": qc.num_clbits,
                            "qasm_sha256": digest,
                            "qasm_path": path,
                            "qasm_status": qasm_status,
                            "generation_seconds": round(time.time() - t0, 3),
                            "status": "ok",
                        })
                    except Exception as exc:
                        failures.append({
                            "source": "MQT_Bench", "level": level_name, "algorithm": algo,
                            "requested_size": n, "error": f"{type(exc).__name__}: {exc}"[:200],
                        })
    return records, failures


def acquire_qasmbench() -> tuple[list[dict], list[dict]]:
    bench_dir = RAW_DIR / "QASMBench"
    records, failures = [], []
    qasm_files = sorted(bench_dir.rglob("*.qasm")) if bench_dir.exists() else []
    for path in qasm_files:
        t0 = time.time()
        try:
            qc = qasm2.loads(path.read_text(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            if qc.num_qubits == 0:
                raise ValueError("empty circuit")
            text = path.read_text()
            digest = sha256_text(text)
            algo = path.stem.split("_")[0] if "_" in path.stem else path.stem
            circuit_id = f"qasmb-{path.stem}-{digest[:10]}"
            records.append({
                "circuit_id": circuit_id,
                "source": "QASMBench",
                "algorithm": algo,
                "abstraction_level": "QASM2",
                "requested_size": qc.num_qubits,
                "rep": 0,
                "num_qubits": qc.num_qubits,
                "num_clbits": qc.num_clbits,
                "qasm_sha256": digest,
                "qasm_path": str(path),
                "qasm_status": "ok",
                "generation_seconds": round(time.time() - t0, 3),
                "status": "ok",
            })
        except Exception as exc:
            failures.append({"source": "QASMBench", "file": path.name,
                             "error": f"{type(exc).__name__}: {exc}"[:200]})
    return records, failures


# ----------------------------------------------------------------------------- features + labels
FORBIDDEN_FOR_BASIS = None


def circuit_features(qc: QuantumCircuit) -> dict:
    core = strip_nonunitary(qc)
    stats = gate_statistics(core)
    depth = core.depth()
    n = core.num_qubits
    gate_counts = stats["gate_counts"]
    probs = np.array(list(gate_counts.values()), dtype=float) if gate_counts else np.array([0.0])
    probs = probs / probs.sum() if probs.sum() else probs
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum()) if gate_counts else 0.0
    feats = {
        "num_qubits": n,
        "num_clbits": qc.num_clbits,
        "gate_count": stats["gate_count"],
        "circuit_depth": depth,
        "one_qubit_gates": stats["one_qubit_gates"],
        "two_qubit_gates": stats["two_qubit_gates"],
        "multi_qubit_gates": stats["multi_qubit_gates"],
        "cx_count": int(gate_counts.get("cx", 0)),
        "t_count_orig": int(gate_counts.get("t", 0)),
        "tdg_count_orig": int(gate_counts.get("tdg", 0)),
        "ccx_count_orig": int(gate_counts.get("ccx", 0)),
        "rz_count_orig": int(gate_counts.get("rz", 0)),
        "measure_count": int(gate_counts.get("measure", 0)),
        "distinct_gate_types": len(gate_counts),
        "gate_type_entropy": entropy,
        "gate_density": stats["gate_count"] / max(1, n * depth),
        "gates_per_layer": stats["gate_count"] / max(1, depth),
        "depth_per_qubit": depth / max(1, n),
    }
    feats.update(interaction_graph_stats(core))
    return feats


def clifford_t_decompose(qc: QuantumCircuit, eps_list: list[float], synth: "AngleSynthesizer") -> tuple[dict, dict]:
    """精确展开到 {rz, sx, x, cx} 基，再按角度给出逐角合成的 Clifford+T 资源。

    返回 (structural_features, {eps: labels})
    """
    base = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0, seed_transpiler=0)
    n = base.num_qubits

    # 逐量子比特记录 rz 及其 T 代价（每个 eps 各一份）
    rz_per_qubit: dict[int, list[tuple[bool, float, dict[float, int]]]] = {i: [] for i in range(n)}
    n_clifford_rz = 0
    n_nonclifford_rz = 0
    t_count_clifford = 0
    clifford_ops = 0
    cx_count = 0
    other_ops: dict[str, int] = {}

    for inst in base.data:
        name = inst.operation.name
        if name in {"measure", "barrier", "reset", "delay"}:
            continue
        idxs = [base.find_bit(q).index for q in inst.qubits]
        if name == "rz":
            theta = float(inst.operation.params[0])
            is_cliff, tcost = angle_class(theta)
            if is_cliff:
                n_clifford_rz += 1
                t_count_clifford += tcost
            else:
                n_nonclifford_rz += 1
            costs = {eps: (tcost if is_cliff else synth.t_cost(theta, eps)) for eps in eps_list}
            if len(idxs) == 1:
                rz_per_qubit[idxs[0]].append((is_cliff, theta, costs))
        elif name in {"sx", "x", "s", "sdg", "h", "z", "sy", "y", "t", "tdg"}:
            clifford_ops += 1
        elif name == "cx":
            cx_count += 1
        else:
            other_ops[name] = other_ops.get(name, 0) + 1

    t_depth_struct = t_structure_depth(base, n)

    structural = {
        "n_clifford_rz": n_clifford_rz,
        "n_nonclifford_rz": n_nonclifford_rz,
        "t_count_clifford_part": t_count_clifford,
        "clifford_ops_after_ct": clifford_ops,
        "cx_after_ct": cx_count,
        "t_depth_structural": t_depth_struct,
        "unexpected_ops_after_ct": json.dumps(other_ops),
    }
    labels = {}
    for eps in eps_list:
        t_count = t_count_clifford + sum(
            rec[2][eps] for gate_list in rz_per_qubit.values() for rec in gate_list if not rec[0]
        )
        # 串行下界：单个量子比特上所有旋转的 T 代价连续排列
        t_depth_serial = max(
            (sum(rec[2][eps] for rec in gate_list) for gate_list in rz_per_qubit.values()),
            default=0,
        )
        noncliff_costs = [rec[2][eps] for gl in rz_per_qubit.values() for rec in gl if not rec[0]]
        labels[eps] = {
            "t_count": int(t_count),
            "t_depth_serial": int(max(t_depth_serial, t_depth_struct)),
            "t_per_rotation": float(np.mean(noncliff_costs)) if noncliff_costs else 0.0,
        }
    return structural, labels


def t_structure_depth(qc: QuantumCircuit, n: int) -> int:
    """T 层深度：把 rz 记为 t，其它门记为同比特上的 id，保留依赖结构后取深度。"""
    circuit = QuantumCircuit(n)
    for inst in qc.data:
        name = inst.operation.name
        if name in {"measure", "barrier", "reset", "delay"}:
            continue
        qubits = [circuit.qubits[qc.find_bit(q).index] for q in inst.qubits]
        if name == "rz":
            if len(qubits) == 1:
                circuit.t(qubits[0])
            else:
                # 受控 rz 已应被展开为 cx+rz；若残留，则视为一个 T 层块
                circuit.t(qubits[0])
        else:
            for q in qubits:
                circuit.id(q)
    return circuit.depth()


class AngleSynthesizer:
    """用 Ross–Selinger（qiskit gridsynth 后端）对单个 RZ 角做 Clifford+T 合成并缓存 T 门数。

    缓存键为 (round(theta, 12), epsilon)，因为同一 ε 下重复角度极多（QFT/QPE 的 π/2^k 序列）。
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[float, float], int] = {}
        self._fallback = 0
        try:
            from qiskit.synthesis.discrete_basis.ross_selinger import gridsynth_rz
            self._gridsynth = gridsynth_rz
        except Exception:
            self._gridsynth = None

    def t_cost(self, theta: float, eps: float) -> int:
        key = (round(theta, 12), float(eps))
        if key in self._cache:
            return self._cache[key]
        if self._gridsynth is None:
            self._fallback += 1
            cost = int(math.ceil(3 * math.log2(1 / eps))) + 1
        else:
            try:
                gate_circuit = self._gridsynth(theta, epsilon=eps)
                cost = int(gate_circuit.count_ops().get("t", 0))
                cost = max(cost, 1)
            except Exception:
                self._fallback += 1
                cost = int(math.ceil(3 * math.log2(1 / eps))) + 1
        self._cache[key] = cost
        return cost

    @property
    def stats(self) -> dict:
        return {"unique_angles": len(self._cache), "fallback_calls": self._fallback}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-circuits", type=int, default=None)
    ap.add_argument("--levels", nargs="+", default=["ALG", "INDEP"])
    ap.add_argument("--eps", nargs="+", type=float, default=[1e-3, 1e-6, 1e-9])
    ap.add_argument("--random-reps", type=int, default=4)
    ap.add_argument("--include-qasmbench", action="store_true")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--extend", action="store_true")
    ap.add_argument("--size-grid", nargs="+", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for d in (QASM_DIR, PROC_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    manifest_path = RAW_DIR / "circuits_manifest.parquet"
    if args.extend and manifest_path.exists():
        # 规模扩展：在既有清单基础上追加新尺寸/新随机种子，用于把可标注线路补齐到 MVP 目标
        print("[1/4] 扩展采集（--extend）...", flush=True)
        base = pd.read_parquet(manifest_path)
        records, failures = acquire_mqt(args.levels, args.max_circuits, args.random_reps, args.seed, args.size_grid)
        df = pd.concat([base, pd.DataFrame(records)], ignore_index=True).drop_duplicates(subset=["qasm_sha256"])
        df.to_parquet(manifest_path, index=False)
        pd.DataFrame(failures).to_csv(LOG_DIR / f"acquisition_failures_extend_{int(time.time())}.csv", index=False)
    elif args.resume and manifest_path.exists():
        print("[1/4] 复用已有线路清单（--resume）", flush=True)
        df = pd.read_parquet(manifest_path).drop_duplicates(subset=["qasm_sha256"])
        failures: list[dict] = []
    else:
        print("[1/4] 采集 MQT Bench 线路 ...", flush=True)
        records, failures = acquire_mqt(args.levels, args.max_circuits, args.random_reps, args.seed, args.size_grid)
        if args.include_qasmbench:
            print("[1/4] 采集 QASMBench 线路 ...", flush=True)
            q_records, q_fail = acquire_qasmbench()
            records.extend(q_records)
            failures.extend(q_fail)

        if not records:
            raise SystemExit("没有采集到任何线路")

        # 去重（同一 qasm 内容只保留一条）
        df = pd.DataFrame(records).drop_duplicates(subset=["qasm_sha256"])
        df.to_parquet(manifest_path, index=False)
        pd.DataFrame(failures).to_csv(LOG_DIR / "acquisition_failures.csv", index=False)
    print(f"    有效线路 {len(df)} 条，失败组合 {len(failures)} 个", flush=True)

    print("[2/4] 初始化 Ross–Selinger 角度合成器 ...", flush=True)
    synth = AngleSynthesizer()
    if synth._gridsynth is None:
        print("    警告：qiskit gridsynth_rz 不可用，回退到 Ross–Selinger 解析上界", flush=True)

    print("[3/4] 提取特征 + Clifford+T 资源标注 ...", flush=True)
    feature_rows, label_rows, errors = [], [], []
    done_ids: set[str] = set()
    if args.resume and (PROC_DIR / "circuit_features.parquet").exists():
        prev = pd.read_parquet(PROC_DIR / "circuit_features.parquet")
        feature_rows = prev.to_dict("records")
        done_ids = set(prev["circuit_id"])
        if (PROC_DIR / "logical_resources.parquet").exists():
            label_rows = pd.read_parquet(PROC_DIR / "logical_resources.parquet").to_dict("records")
        print(f"    断点续跑：已完成 {len(done_ids)} 条", flush=True)

    for i, row in enumerate(df.itertuples(), start=1):
        if row.circuit_id in done_ids:
            continue
        try:
            if not row.qasm_path or not Path(row.qasm_path).exists():
                raise ValueError("no_qasm_file: 既无 QASM2 也无 QPY 落盘（见 scripts/acquire_dynamic.py）")
            # ALG 层是黑盒算子级线路，其 opaque 定义的展开规模随 n 爆炸（grover n≥16 单条即不可完成）；
            # 只保留 n ≤ ALG_LABEL_MAX_SIZE 的 ALG 线路做标注，更大规模由 INDEP 层显式线路承担。
            if row.abstraction_level == "ALG" and int(row.requested_size) > ALG_LABEL_MAX_SIZE:
                raise ValueError(f"skipped_expansion_cost: ALG n={row.requested_size} > {ALG_LABEL_MAX_SIZE}")
            with time_limit(args.timeout):
                qc = load_circuit(row.qasm_path)
                feats = circuit_features(qc)
                structural, labels = clifford_t_decompose(qc, args.eps, synth)
            feats.update(structural)
            feats.update({k: row._asdict()[k] for k in
                          ["circuit_id", "source", "algorithm", "abstraction_level", "requested_size", "num_clbits"]})
            feats["num_clbits"] = row.num_clbits
            feature_rows.append(feats)
            for eps, lab in labels.items():
                label_rows.append({
                    "circuit_id": row.circuit_id,
                    "algorithm": row.algorithm,
                    "source": row.source,
                    "abstraction_level": row.abstraction_level,
                    "num_qubits": feats["num_qubits"],
                    "synthesis_epsilon": eps,
                    "logical_t_count": lab["t_count"],
                    "logical_t_depth": lab["t_depth_serial"],
                    "t_depth_structural": feats["t_depth_structural"],
                    "t_per_rotation": lab["t_per_rotation"],
                    "logical_qubits": feats["num_qubits"],
                    "clifford_count": feats["clifford_ops_after_ct"],
                    "cx_count": feats["cx_after_ct"],
                })
        except CircuitTimeout as exc:
            errors.append({"circuit_id": row.circuit_id, "error": f"timeout: {exc}"})
        except Exception as exc:
            errors.append({"circuit_id": row.circuit_id, "error": f"{type(exc).__name__}: {exc}"[:200]})
        if i % 50 == 0:
            print(f"    processed {i}/{len(df)}  ({time.time()-t_start:.0f}s)  errors={len(errors)}", flush=True)
            pd.DataFrame(feature_rows).to_parquet(PROC_DIR / "circuit_features.parquet", index=False)
            if label_rows:
                pd.DataFrame(label_rows).to_parquet(PROC_DIR / "logical_resources.parquet", index=False)

    features_df = pd.DataFrame(feature_rows)
    labels_df = pd.DataFrame(label_rows)
    features_df.to_parquet(PROC_DIR / "circuit_features.parquet", index=False)
    labels_df.to_parquet(PROC_DIR / "logical_resources.parquet", index=False)
    pd.DataFrame(errors).to_csv(LOG_DIR / "feature_errors.csv", index=False)

    print("[4/4] 完成", flush=True)
    with open(PROC_DIR / "synthesis_calibration.json", "w") as f:
        json.dump({"backend": "qiskit.synthesis.discrete_basis.ross_selinger.gridsynth_rz",
                   "stats": synth.stats}, f, indent=2)
    print(f"    角度合成器：unique_angles={synth.stats['unique_angles']} fallback={synth.stats['fallback_calls']}")
    print(f"    circuits={features_df['circuit_id'].nunique()} features={features_df.shape[1]} "
          f"label_rows={len(labels_df)} errors={len(errors)} elapsed={time.time()-t_start:.0f}s")
    if len(features_df):
        print(f"    qubits: {features_df.num_qubits.min()}–{features_df.num_qubits.max()}, "
              f"algorithms: {features_df.algorithm.nunique()}, sources: {features_df.source.unique().tolist()}")


if __name__ == "__main__":
    main()
