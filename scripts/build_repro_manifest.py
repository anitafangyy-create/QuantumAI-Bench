#!/usr/bin/env python3
"""Build the reproducibility package: SHA-256 manifest, dataset version record, environment freeze.

Writes into reproducibility/ so that a reviewer can (a) verify that the artefacts they received are
bit-identical to the ones the manuscript reports, and (b) see the exact library versions used.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproducibility"

INCLUDE = [
    "raw_data/circuits_manifest.parquet",
    "processed_data/circuit_features.parquet",
    "processed_data/logical_resources.parquet",
    "processed_data/fault_tolerant_resources.parquet",
    "results/evaluation",
    "results/figures",
    "scripts",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> list[Path]:
    files = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.is_file() and not q.name.startswith(".")))
    return files


def main() -> None:
    OUT.mkdir(exist_ok=True)
    files = collect()
    lines = [f"{sha256(p)}  {p.relative_to(ROOT)}" for p in files]
    (OUT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

    import pandas as pd
    ft = pd.read_parquet(ROOT / "processed_data" / "fault_tolerant_resources.parquet")
    lab = pd.read_parquet(ROOT / "processed_data" / "logical_resources.parquet")
    feat = pd.read_parquet(ROOT / "processed_data" / "circuit_features.parquet")
    version = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "circuits": int(feat["circuit_id"].nunique()),
        "logical_records": int(len(lab)),
        "physical_records": int(len(ft)),
        "synthesis_epsilon": sorted(ft["synthesis_epsilon"].unique().tolist()),
        "physical_error_rate": sorted(ft["physical_error_rate"].unique().tolist()),
        "logical_error_budget": sorted(ft["logical_error_budget"].unique().tolist()),
        "configs_per_circuit": float(len(ft) / ft["circuit_id"].nunique()),
        "n_files_hashed": len(files),
    }
    (OUT / "DATASET_VERSION.json").write_text(json.dumps(version, indent=2))

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (OUT / "ENVIRONMENT.txt").write_text(
        f"python {platform.python_version()} on {platform.platform()}\n"
        f"executable {sys.executable}\n\n{freeze}"
    )
    print(f"hashed {len(files)} files -> {OUT/'MANIFEST.sha256'}")
    print(json.dumps(version, indent=2))


if __name__ == "__main__":
    main()
