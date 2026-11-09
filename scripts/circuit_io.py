"""统一的线路读取入口：QASM2 / QASM3 / QPY 三种落盘格式。

存在三种格式的唯一原因是 MQT Bench 的部分基准线路带经典条件分支（`dynamic_qft`、
`ghz_dynamic`、`ae`、`iqpe`），qiskit 的 OpenQASM 2 导出器直接拒绝（QASM2ExportError），
QASM 3 也只能救回其中两族（`ghz_dynamic` 的条件含 XOR、`ae` 导出 TypeError）。
于是这类线路改用 qiskit 原生 QPY 保存——它是线路对象的精确序列化，不经过门集重写，
读回逐门一致（见 `scripts/acquire_dynamic.py` 的自检）。

用法：
    from circuit_io import load_circuit
    qc = load_circuit(row.qasm_path)
"""
from __future__ import annotations

from pathlib import Path

from qiskit import qasm2, qasm3, qpy


def load_circuit(path: str | Path):
    """按扩展名/文件头选择解析器，返回单个 QuantumCircuit。"""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".qpy":
        with open(p, "rb") as f:
            obj = qpy.load(f)
        # qiskit 2.x 返回 list；更早版本返回单个对象
        return obj[0] if isinstance(obj, list) else obj

    text = p.read_text()
    if suffix == ".qasm3" or text.lstrip().startswith("OPENQASM 3"):
        return qasm3.loads(text)
    return qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
