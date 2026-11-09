# QuantumAI-FTQC

**A three-layer benchmark and cross-distribution study for fault-tolerant quantum resource estimation.**

This repository contains the code, label pipelines, audit scripts and aggregate results behind the study
*"How Far Can Circuit Structure Predict Fault-Tolerant Quantum Resources?"*. It is deliberately
data-first: every number in the paper can be regenerated from the scripts here, and every label is
accompanied by the invariants used to check it.

The manuscript itself is **not** part of this repository; it is maintained separately so that a code
release never carries a draft paper.

---

## What is here

| Layer | Content | Scale |
| --- | --- | --- |
| Circuit descriptors | 28 structural features per circuit (topology, gate counts, post-decomposition quantities) | 7,502 circuits, 192 algorithm labels, 2–60 qubits |
| Logical resources | T-count, T-depth, logical qubits from Clifford+T synthesis at three precisions | 22,506 records |
| Physical resources | Surface-code proxy over a 3 × 3 × 3 grid (epsilon, p, budget) | 202,554 records, all feasible |

Two public corpora are combined: **MQT Bench** (1,246 circuits, training side, plus 397 same-family
random-circuit instances at new seeds) and **QCircuitBench** (6,236 circuits, test-only cross-source
corpus). Labeling coverage is 98.0%; the 74 circuits with mid-circuit measurement and classical feedback
are excluded by definition because their T-count has no single value.

## Repository layout

```
scripts/            pipeline: acquisition, labeling, features, models, audits, figures
docs/               dataset card, stage report, Chinese README, migration notes, cross-check report
results/
  evaluation/       aggregate metrics, statistical tests, ablations, distribution audits (CSV/JSON)
  figures/          figures 1-7 exactly as plotted in the paper
reproducibility/    SHA-256 manifest, dataset version record, environment freeze
requirements.txt    pinned environment (Python 3.12)
```

Raw circuits (1.9 GB) and derived parquet tables (186 MB) are **not** committed — see *Data* below.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pygridsynth` is required for labeling; `mqt-bench` is required only to (re)acquire the benchmark
circuits. Training uses scikit-learn's histogram gradient boosting, so no GPU is needed: the full
four-protocol sweep runs in under ten minutes on eight CPU cores.

## Data

The repository ships the code and the aggregate results, not the circuits. To rebuild the dataset:

```bash
python scripts/build_dataset.py        # acquire, parse, label: needs network access to the corpora
python scripts/qec_resources.py        # physical layer over the 27-point grid
python scripts/verify_labels.py        # four structural invariants
python scripts/check_eps_scaling.py    # physical scaling invariant
```

Acquisition writes `raw_data/circuits_manifest.parquet` and `raw_data/circuits/*.qasm|*.qpy`; labeling
writes `processed_data/*.parquet`. Both paths are git-ignored. The circuits themselves remain subject to
the licences of MQT Bench and QCircuitBench.

## Reproduce

```bash
bash scripts/reproduce_all.sh                # everything: labels -> models -> audits -> figures (~hours)
STAGE=verify bash scripts/reproduce_all.sh   # checks + audits only (~minutes)
```

The full path runs the four protocols (grouped, leave-one-algorithm-out, size extrapolation, cross-source)
over three model families, the two targeted-removal ablations, the feature-tier ablation, the zero-target
audit, the distribution-shift audit, the statistical tests and the figures.

## Headline numbers

Median relative error (MdAPE%) on T-count, random forest / gradient-boosted trees:

| Protocol | Training → test | RF | GBDT |
| --- | --- | --- | --- |
| Grouped (Protocol A) | circuit-level 70/30 split | 0.01 | 0.19 |
| Leave-one-algorithm-out (B) | 28 families held out | 0.79 | 0.86 |
| Size extrapolation (C) | train ≤ 16 qubits, test > 16 | 2.68 | 3.10 |
| Cross-source (D) | MQT Bench → QCircuitBench | 28.31 | 24.19 |

Three findings are worth reading before using these numbers:

1. **In-distribution accuracy is interpolation, not generalization.** With only synthesis-free
   descriptors the grouped forest already reports a sub-0.01% median error, because near-duplicate
   structural vectors occur on both sides of the split.
2. **The cross-source gain is carried by decomposition-derived features.** Removing the six
   post-decomposition descriptors raises cross-source T-count error from 28.3% to 108.9% (RF) and from
   24.2% to 72.6% (GBDT). Those descriptors cost about 0.6 CPU-second per circuit and precision, i.e. as
   much as the labels they help predict. The surrogate therefore saves repeated configuration inference,
   not decomposition — against the analytic proxy used here it has no wall-clock advantage.
3. **The main transfer obstacle is a convention, not a learning problem.** The two corpora disagree on
   whether rotations arrive pre-decomposed (median rz-gate count 0 vs 1,024, KS = 0.86) and share no
   algorithm labels. Basis canonicalization is the cheapest next experiment.

Zeros are handled explicitly: targets are modeled as `log1p`, relative errors are only computed against
positive references, and the zero-T-count class is scored separately (F1 = 1.000 in-distribution, 0.987
cross-source for GBDT, against 0.034 for ridge).

## Scope and limitations

- Training is single-source (MQT Bench); QCircuitBench is never used for fitting.
- The physical layer is an analytic surface-code proxy with stated calibration constants
  (`A = 0.03`, `p_th = 1e-2`, 15-patch factories, 1 µs cycle). It is not a layout-aware compiler and its
  absolute numbers should not be compared with a full-stack estimator without re-deriving the assumptions.
- Protocol A is repeated over five seeds; B–D use deterministic splits. No seed-stability claim is made
  for B–D.
- The boosted backend is scikit-learn's histogram GBDT rather than native XGBoost (no libomp on the
  development machine); backend identity is recorded per run and never mixed.
- Adaptive circuits whose cost depends on measurement outcomes are out of scope by construction.

## Reproducibility record

`reproducibility/MANIFEST.sha256` hashes the label tables, aggregate metrics, figures and scripts;
`DATASET_VERSION.json` records the dataset composition and the configuration grid; `ENVIRONMENT.txt`
records the exact library versions. Regenerate all three with `scripts/build_repro_manifest.py`.

## Documentation

- [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) — provenance, inclusion rules, label definitions
- [`docs/STAGE1_REPORT.md`](docs/STAGE1_REPORT.md) — stage-one experiments and open gaps
- [`docs/XCHECK_INDEPENDENT_SYNTHESIS.md`](docs/XCHECK_INDEPENDENT_SYNTHESIS.md) — independent synthesizer cross-check
- [`docs/README.zh-CN.md`](docs/README.zh-CN.md) — Chinese overview of the same pipeline

## License

Released for review and reproduction. The licence under which this code may be redistributed is to be
fixed before public announcement (see `LICENSE`); until then, treat it as all rights reserved.

## Contact

Issues and reproducibility questions are welcome through the issue tracker of this repository.
