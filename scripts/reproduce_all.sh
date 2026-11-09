#!/usr/bin/env bash
# One-command regeneration of every table and figure reported in the manuscript.
# Usage: bash scripts/reproduce_all.sh            (full pipeline, hours on 8 cores)
#        STAGE=verify bash scripts/reproduce_all.sh  (labels + audits only, minutes)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
STAGE=${STAGE:-full}

echo "[1/9] label self-checks"
$PY scripts/verify_labels.py

echo "[2/9] dataset statistics"
$PY scripts/report_stats.py

echo "[3/9] epsilon-scaling audit"
$PY scripts/check_eps_scaling.py

if [ "$STAGE" = "verify" ]; then
  echo "[4/9] TQE P0 audits (feature ablation, zero handling, distribution shift)"
  $PY scripts/tqe_p0_audits.py
  echo "[5/9] reproducibility manifest"
  $PY scripts/build_repro_manifest.py
  exit 0
fi

echo "[4/9] four protocols, five seeds, headline configuration"
$PY scripts/train_models.py --protocols grouped loso scale crosssrc --models ridge rf xgb --tag v8

echo "[5/9] targeted-removal ablations"
$PY scripts/train_models.py --protocols crosssrc --models rf xgb --tag v8nc --drop-clifford-train
$PY scripts/train_models.py --protocols crosssrc --models rf xgb --tag v8ndyn \
    --drop-algos-train ae dynamic_qft ghz_dynamic iqpe

echo "[6/9] statistical tests and seed variance"
$PY scripts/stats_tests.py --bootstrap 2000
$PY scripts/seed_variance.py

echo "[7/9] TQE P0 audits"
$PY scripts/tqe_p0_audits.py

echo "[8/9] figures"
$PY scripts/make_figures.py
$PY scripts/make_figure7.py

echo "[9/9] reproducibility manifest"
$PY scripts/build_repro_manifest.py
echo "done — the manuscript itself is maintained outside this repository"
