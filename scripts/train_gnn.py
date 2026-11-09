#!/usr/bin/env python3
"""GNN 基线：直接用线路的门级 DAG 预测逻辑 T-count（对应消融 E1 的"图结构是否有用"）。

与 `train_models.py` 的表格特征模型对照：
  - 表格模型吃的是人工结构特征（门计数、深度、连通度…），GNN 吃的是门级数据流图本身；
  - 因此两者在同一 grouped 划分下的差异，就是"图结构"相对"人工特征"的增量。

图构造（每个门一个节点）：
  - 节点特征：门名 one-hot（取语料里最高频的 20 种 + other）、门的比特数、归一化位置、
    参数角的 sin/cos（无参数为 0）、涉及比特均值（归一化）。
  - 边：同一比特上"前一个门 → 当前门"，双向各一条。不做跨比特全连接（保持 DAG 的稀疏性）。
  - 图级特征：log2 比特数、log1p 深度、log1p 门数、log10 合成精度、CX 占比。

模型：3 层 mean-aggregation 消息传递 + 残差 + LayerNorm，节点池化（mean ⊕ max）后接 MLP 回归
log10(T-count + 1)。GPU 训练，15% 验证早停。

用法：
  python scripts/train_gnn.py --protocols grouped crosssrc --epochs 80
产出：
  results/evaluation/gnn_metrics.csv
  results/evaluation/gnn_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
EVAL = ROOT / "results" / "evaluation"
QASM_DIR = ROOT / "raw_data" / "circuits"
CACHE = PROC / "gnn_cache"
EVAL.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

TOPK_GATES = 20
GLOBAL_DIM = 5


# ------------------------------------------------------------------ 图构建
def _load_qasm(path):
    """统一解析入口：MQT 的 qasm 里含 rxx/rzz 等旧 qelib 门，标准 OpenQASM2 会拒收。"""
    from qiskit import qasm2

    txt = path.read_text()
    try:
        return qasm2.loads(txt)
    except Exception:
        return qasm2.loads(txt, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def gate_vocab(circuit_ids: list[str], n_sample: int = 400) -> dict[str, int]:
    from qiskit import qasm2
    cnt: dict[str, int] = {}
    rng = np.random.default_rng(0)
    for cid in rng.choice(circuit_ids, size=min(n_sample, len(circuit_ids)), replace=False):
        p = QASM_DIR / f"{cid}.qasm"
        if not p.exists():
            continue
        try:
            qc = _load_qasm(p)
        except Exception:
            continue
        for ins in qc.data:
            nm = ins.operation.name
            cnt[nm] = cnt.get(nm, 0) + 1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:TOPK_GATES]
    return {nm: i for i, (nm, _) in enumerate(top)}


def build_graph(cid: str, vocab: dict[str, int]):
    """返回 (x[V,F], edge_index[2,E], n_qubits, depth, n_gates, cx_ratio)。"""
    from qiskit import qasm2

    p = QASM_DIR / f"{cid}.qasm"
    qc = _load_qasm(p)
    n = max(int(qc.num_qubits), 1)
    ops = qc.data
    m = len(ops)
    if m == 0:
        return None
    V = len(vocab) + 1
    F = V + 5
    x = np.zeros((m, F), dtype=np.float32)
    last_on_qubit = [-1] * n
    src, dst = [], []
    cx = 0
    for i, ins in enumerate(ops):
        op = ins.operation
        nm = op.name
        qi = [qc.find_bit(q).index for q in ins.qubits]
        j = vocab.get(nm, len(vocab))
        x[i, j] = 1.0
        x[i, V + 0] = math.log1p(len(qi)) / 2.0
        x[i, V + 1] = i / max(m - 1, 1)
        ang = float(op.params[0]) if getattr(op, "params", None) and len(op.params) else 0.0
        x[i, V + 2] = math.sin(ang)
        x[i, V + 3] = math.cos(ang) - 1.0
        x[i, V + 4] = float(np.mean(qi)) / n
        if nm in ("cx", "cnot", "ecr", "cz", "swap"):
            cx += 1
        for q in qi:
            if last_on_qubit[q] >= 0:
                src.append(last_on_qubit[q]); dst.append(i)
                src.append(i); dst.append(last_on_qubit[q])
            last_on_qubit[q] = i
    if not src:
        src, dst = [0], [0]
    edge_index = np.array([src, dst], dtype=np.int64)
    return x, edge_index, n, int(qc.depth()), m, cx / max(m, 1)


def load_or_build(cid: str, vocab: dict[str, int]):
    f = CACHE / f"{cid}.npz"
    if f.exists():
        try:
            z = np.load(f)
            return z["x"], z["e"], float(z["g"][0]), float(z["g"][1]), float(z["g"][2]), float(z["g"][3])
        except Exception:
            pass
    g = build_graph(cid, vocab)
    if g is None:
        return None
    x, e, n, d, m, cxr = g
    np.savez_compressed(f, x=x, e=e, g=np.array([n, d, m, cxr], dtype=np.float32))
    return x, e, float(n), float(d), float(m), cxr


# ------------------------------------------------------------------ 模型
def make_model(dim_in: int = 0):
    import torch
    import torch.nn as nn

    class MP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.msg = nn.Linear(d, d)
            self.upd = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
            self.ln = nn.LayerNorm(d)

        def forward(self, h, ei):
            src, dst = ei[0], ei[1]
            m = self.msg(h)[src]
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            deg = torch.zeros(h.size(0), device=h.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            agg = agg / deg.clamp(min=1).unsqueeze(1)
            return self.ln(h + self.upd(agg))

    class GNN(nn.Module):
        def __init__(self, n_gate_vocab: int, d=64):
            super().__init__()
            self.emb = nn.Sequential(nn.Linear(n_gate_vocab + 5, d), nn.ReLU(), nn.Linear(d, d))
            self.layers = nn.ModuleList([MP(d) for _ in range(3)])
            self.head = nn.Sequential(nn.Linear(2 * d + GLOBAL_DIM, d), nn.ReLU(),
                                      nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, xs, eis, batch, gfeat):
            h = self.emb(xs)
            for L in self.layers:
                h = L(h, eis)
            # 池化：按图求 mean 与 max
            B = gfeat.size(0)
            mean = torch.zeros(B, h.size(1), device=h.device).index_add_(0, batch, h)
            cnt = torch.zeros(B, device=h.device).index_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
            mean = mean / cnt.clamp(min=1).unsqueeze(1)
            mx = torch.full((B, h.size(1)), -1e9, device=h.device)
            mx = mx.scatter_reduce(0, batch.unsqueeze(1).expand_as(h), h, reduce="amax", include_self=True)
            z = torch.cat([mean, mx, gfeat], dim=1)
            return self.head(z).squeeze(1)

    return GNN


# ------------------------------------------------------------------ 指标
def metrics(y_true: np.ndarray, pred_log: np.ndarray) -> dict:
    from sklearn.metrics import r2_score
    y_log = np.log10(y_true + 1.0)
    pred = 10 ** pred_log - 1.0
    pred = np.clip(pred, 0, None)
    pos = y_true > 0
    l10 = np.abs(np.log10(np.maximum(pred[pos] / y_true[pos], 1e-300)) if pos.sum() else np.array([0.0]))
    return {
        "r2_log": float(r2_score(y_log, pred_log)),
        "mdape_pct": float(np.median(np.minimum(l10 * 100, 1e6))) if pos.sum() else float("nan"),
        "frac_gt_10x": float(np.mean(l10 > 1.0)) if pos.sum() else float("nan"),
        "n": int(len(y_true)), "n_nonzero_ref": int(pos.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["grouped", "crosssrc"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-gates", type=int, default=20000, help="超此门数的线路跳过（图太大，显存与时间不划算）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target", default="logical_t_count")
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--external-source", default="qcircuitbench")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gnn] device={dev}", flush=True)

    lab = pd.read_parquet(PROC / "logical_resources.parquet")
    lab = lab[(lab.synthesis_epsilon == args.eps) & lab[args.target].notna()]
    feat = pd.read_parquet(PROC / "circuit_features.parquet")[
        ["circuit_id", "num_qubits", "circuit_depth", "gate_count", "cx_count", "algorithm", "source"]
    ]
    df = lab[["circuit_id", args.target]].merge(feat, on="circuit_id", how="inner")
    df = df[df.gate_count <= args.max_gates].reset_index(drop=True)
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=0).reset_index(drop=True)
    print(f"[gnn] 候选线路 {len(df)}（门数≤{args.max_gates}）", flush=True)

    vocab = gate_vocab(df.circuit_id.tolist())
    print(f"[gnn] 门表 {len(vocab)} 种：{list(vocab)[:8]}…", flush=True)

    # 先构图（带磁盘缓存），失败的线路直接剔除
    graphs, keep = [], []
    t0 = time.time()
    for i, (cid, y) in enumerate(zip(df.circuit_id, df[args.target].to_numpy())):
        g = load_or_build(cid, vocab)
        if g is None:
            continue
        x, e, n, d, m, cxr = g
        gfeat = np.array([math.log2(n + 1) / 6, math.log1p(d) / 10, math.log1p(m) / 10,
                          math.log10(args.eps) / 9, cxr], dtype=np.float32)
        graphs.append((x, e, gfeat))
        keep.append(i)
        if (i + 1) % 500 == 0:
            print(f"  图 {i+1}/{len(df)}（{(time.time()-t0)/60:.1f} min）", flush=True)
    df = df.iloc[keep].reset_index(drop=True)
    print(f"[gnn] 构图完成 {len(graphs)} 条，用时 {(time.time()-t0)/60:.1f} min", flush=True)

    y = np.log10(df[args.target].to_numpy() + 1.0).astype(np.float32)
    all_rows, pred_rows = [], []

    def split(protocol: str):
        rng = np.random.default_rng(42)
        if protocol == "grouped":
            algos = df.algorithm.unique()
            rng.shuffle(algos)
            n_tr = max(1, int(0.7 * len(algos)))
            n_va = max(1, int(0.15 * len(algos)))
            tr_a = set(algos[:n_tr]); va_a = set(algos[n_tr:n_tr + n_va])
            tr = df.index[df.algorithm.isin(tr_a)].to_numpy()
            va = df.index[df.algorithm.isin(va_a)].to_numpy()
            te = df.index[~df.algorithm.isin(tr_a | va_a)].to_numpy()
            return tr, va, te
        if protocol == "crosssrc":
            tr = df.index[df.source != args.external_source].to_numpy()
            te = df.index[df.source == args.external_source].to_numpy()
            va = tr[rng.choice(len(tr), size=max(1, int(0.1 * len(tr))), replace=False)] if len(tr) else tr
            return tr, va, te
        raise ValueError(protocol)

    for protocol in args.protocols:
        tr, va, te = split(protocol)
        if len(tr) < 20 or len(te) < 5:
            print(f"[gnn] {protocol}: 划分过小（train={len(tr)} test={len(te)}），跳过", flush=True)
            continue
        print(f"[gnn] {protocol}: train={len(tr)} val={len(va)} test={len(te)}", flush=True)
        GNN = make_model()
        model = GNN(len(vocab) + 1).to(dev)  # +1 = 'other' 槽位，节点特征维 = len(vocab)+1+5
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lossf = nn.MSELoss()
        best, best_state = 1e9, None

        def batch_tensors(idx):
            xs, eis, bs, gf = [], [], [], []
            off = 0
            for b, i in enumerate(idx):
                x, e, g = graphs[i]
                xs.append(torch.from_numpy(x))
                eis.append(torch.from_numpy(e).clone() + off)
                bs.append(torch.full((x.shape[0],), b, dtype=torch.long))
                gf.append(torch.from_numpy(g))
                off += x.shape[0]
            return (torch.cat(xs).to(dev), torch.cat(eis, dim=1).to(dev),
                    torch.cat(bs).to(dev), torch.stack(gf).to(dev))

        for ep in range(1, args.epochs + 1):
            model.train()
            order = np.random.default_rng(ep).permutation(tr)
            tot, nb = 0.0, 0
            for s in range(0, len(order), args.batch):
                bi = order[s:s + args.batch]
                xs, eis, bs, gf = batch_tensors(bi)
                pred = model(xs, eis, bs, gf)
                loss = lossf(pred, torch.from_numpy(y[bi]).to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            model.eval()
            with torch.no_grad():
                pv = []
                for s in range(0, len(va), 256):
                    bi = va[s:s + 256]
                    xs, eis, bs, gf = batch_tensors(bi)
                    pv.append(model(xs, eis, bs, gf).cpu().numpy())
                vloss = float(np.mean((np.concatenate(pv) - y[va]) ** 2)) if len(va) else 0.0
            if vloss < best:
                best, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 10 == 0 or ep == 1:
                print(f"  ep{ep:3d} train_mse={tot/max(nb,1):.4f} val_mse={vloss:.4f}", flush=True)
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            pt = []
            for s in range(0, len(te), 256):
                bi = te[s:s + 256]
                xs, eis, bs, gf = batch_tensors(bi)
                pt.append(model(xs, eis, bs, gf).cpu().numpy())
            pred_log = np.concatenate(pt)
        y_true = df[args.target].to_numpy()[te]
        m = metrics(y_true, pred_log)
        row = {"protocol": protocol, "model": "gnn", "target": args.target, "seed": 0, **m}
        all_rows.append(row)
        print(f"[gnn] {protocol} 结果：{json.dumps(m, ensure_ascii=False)}", flush=True)
        for i, pl in zip(te, pred_log):
            pred_rows.append({"protocol": protocol, "circuit_id": df.circuit_id.iloc[i],
                              "algorithm": df.algorithm.iloc[i], "source": df.source.iloc[i],
                              "y_true": float(df[args.target].iloc[i]), "pred_log10": float(pl)})

    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(all_rows).to_csv(EVAL / f"gnn_metrics_{tag}.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(EVAL / f"gnn_predictions_{tag}.csv", index=False)
    print(f"[gnn] 写出 gnn_metrics_{tag}.csv / gnn_predictions_{tag}.csv", flush=True)


if __name__ == "__main__":
    main()
