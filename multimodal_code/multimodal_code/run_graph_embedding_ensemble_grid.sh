#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Fast graph-embedding ensemble grid for addH/addH-2 -> addH-out
# Run from: /data/home/terminator/RL/multi-view
# Uses existing master tables and dual graph embeddings.
# Does NOT run FAIR-Chem, CLIP, or RoBERTa.
# ============================================================

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
cd "$ROOT"

PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
SRC_OUT="${SRC_OUT:-$ROOT/outputs_addh_full_mm_envsplit}"
GRID_ROOT="${GRID_ROOT:-$ROOT/outputs_addh_graph_ensemble}"

ADDH_MASTER="${ADDH_MASTER:-$SRC_OUT/addH_master_target_weighted_mild.csv}"
ADDH_DUAL_EMB="${ADDH_DUAL_EMB:-$SRC_OUT/addH_dual_eq_emb.pkl}"
ADDHOUT_MASTER="${ADDHOUT_MASTER:-$SRC_OUT/addH_out_master_normalized.csv}"
ADDHOUT_DUAL_EMB="${ADDHOUT_DUAL_EMB:-$SRC_OUT/addH_out_dual_eq_emb.pkl}"

SEEDS="${SEEDS:-42,52,62,72,82}"
EXPS_TO_RUN="${EXPS_TO_RUN:-all}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
N_SPLITS="${N_SPLITS:-4}"
VAL_FRAC="${VAL_FRAC:-0.25}"

mkdir -p "$GRID_ROOT"

# name | target_abs_max | feature_mode | pca_dim | models | max_val_mae | max_test_rmse | max_abs_pred | stacking(1/0)
COMMON_MODELS="ridge,huber,elasticnet,extratrees,rf,hgb,gbrt,mlp,catboost"
FAST_MODELS="ridge,huber,elasticnet,extratrees,hgb,gbrt,mlp_small,catboost"
TREE_MODELS="extratrees,extratrees_deep,rf,hgb,gbrt,catboost,xgboost,lightgbm"
LINEAR_MODELS="ridge,ridge_strong,huber,elasticnet,mlp_small"

EXPERIMENTS=(
"ge00_t10_full_common_stack|10|full|0|$COMMON_MODELS|3.0|4.0|10|1"
"ge01_t10_full_fast_stack|10|full|0|$FAST_MODELS|3.0|4.0|10|1"
"ge02_t10_full_tree_stack|10|full|0|$TREE_MODELS|3.0|4.0|10|1"
"ge03_t10_full_linear_stack|10|full|0|$LINEAR_MODELS|3.0|4.0|10|1"
"ge04_t10_delta_common_stack|10|delta|0|$COMMON_MODELS|3.0|4.0|10|1"
"ge05_t10_addh_common_stack|10|addh|0|$COMMON_MODELS|3.0|4.0|10|1"
"ge06_t10_bare_common_stack|10|bare|0|$COMMON_MODELS|3.0|4.0|10|1"
"ge07_t10_addh_delta_common_stack|10|addh_delta|0|$COMMON_MODELS|3.0|4.0|10|1"
"ge08_t10_bare_delta_common_stack|10|bare_delta|0|$COMMON_MODELS|3.0|4.0|10|1"
"ge09_t9_full_common_stack|9|full|0|$COMMON_MODELS|3.0|3.8|9|1"
"ge10_t8_full_common_stack|8|full|0|$COMMON_MODELS|3.0|3.5|8|1"
"ge11_t7_full_common_stack|7|full|0|$COMMON_MODELS|2.8|3.2|7|1"
"ge12_t10_full_pca64_stack|10|full|64|$COMMON_MODELS|3.0|4.0|10|1"
"ge13_t10_full_pca128_stack|10|full|128|$COMMON_MODELS|3.0|4.0|10|1"
"ge14_t10_delta_pca64_stack|10|delta|64|$COMMON_MODELS|3.0|4.0|10|1"
"ge15_t10_addh_delta_pca128_stack|10|addh_delta|128|$COMMON_MODELS|3.0|4.0|10|1"
"ge16_t9_delta_common_stack|9|delta|0|$COMMON_MODELS|3.0|3.8|9|1"
"ge17_t8_delta_common_stack|8|delta|0|$COMMON_MODELS|3.0|3.5|8|1"
"ge18_t10_full_common_nostack|10|full|0|$COMMON_MODELS|3.0|4.0|10|0"
"ge19_t10_delta_common_nostack|10|delta|0|$COMMON_MODELS|3.0|4.0|10|0"
)

should_run() {
  local name="$1"
  if [[ "$EXPS_TO_RUN" == "all" ]]; then return 0; fi
  IFS=',' read -ra arr <<< "$EXPS_TO_RUN"
  for x in "${arr[@]}"; do [[ "$x" == "$name" ]] && return 0; done
  return 1
}

run_one() {
  local spec="$1"
  IFS='|' read -r NAME TABS FMODE PCA MODELS MAXVAL MAXTEST MAXPRED STACK <<< "$spec"
  if ! should_run "$NAME"; then echo "[SKIP] $NAME"; return 0; fi

  local EXP_DIR="$GRID_ROOT/$NAME"
  local CV_DIR="$EXP_DIR/cv_data"
  local WORK_DIR="$EXP_DIR/work"
  local LOG_DIR="$EXP_DIR/logs"
  mkdir -p "$EXP_DIR" "$LOG_DIR"

  if [[ "$SKIP_COMPLETED" == "1" && "$FORCE_RERUN" != "1" && -f "$WORK_DIR/test_oof_graph_ensemble_metrics.json" && -f "$WORK_DIR/addH_out_graph_ensemble_by_id.csv" ]]; then
    echo "[DONE/SKIP] $NAME already completed"
    return 0
  fi

  if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -rf "$CV_DIR" "$WORK_DIR"
  fi
  mkdir -p "$CV_DIR" "$WORK_DIR"

  echo "============================================================"
  echo "[EXP] $NAME"
  echo " target_abs_max=$TABS feature_mode=$FMODE pca_dim=$PCA stack=$STACK"
  echo " models=$MODELS"
  echo "============================================================"

  cat > "$EXP_DIR/experiment_config.json" <<EOF
{
  "name": "$NAME",
  "target_abs_max": $TABS,
  "feature_mode": "$FMODE",
  "pca_dim": $PCA,
  "models": "$MODELS",
  "seeds": "$SEEDS",
  "enable_stacking": $STACK
}
EOF

  if [[ ! -f "$CV_DIR/fold_0/regress_train.pkl" ]]; then
    echo "[STEP A] Build CV pkl for graph ensemble..."
    "$PY_MM" 04_make_multiview_data_cv_multimodal.py \
      --addh-master-csv "$ADDH_MASTER" \
      --eq-emb-pkl "$ADDH_DUAL_EMB" \
      --addhout-master-csv "$ADDHOUT_MASTER" \
      --addhout-eq-emb-pkl "$ADDHOUT_DUAL_EMB" \
      --out-dir "$CV_DIR" \
      --group-col family_base_miller \
      --cv-mode stratified_gkfold \
      --n-splits "$N_SPLITS" \
      --val-mode stratified_group_holdout \
      --val-frac "$VAL_FRAC" \
      --seed 42 \
      --include-regress-eq-emb \
      --include-addhout-in-clip \
      --exclude-target-outliers \
      --target-abs-max "$TABS" \
      --stratify-bins 5 \
      > "$LOG_DIR/build_cv.log" 2>&1
  else
    echo "[STEP A] Reuse existing CV data: $CV_DIR"
  fi

  echo "[STEP B] Train graph embedding ensemble..."
  STACK_ARG=()
  [[ "$STACK" == "1" ]] && STACK_ARG=(--enable-stacking)
  "$PY_MM" 13_train_graph_embedding_ensemble.py \
    --cv-root "$CV_DIR" \
    --work-dir "$WORK_DIR" \
    --models "$MODELS" \
    --seeds "$SEEDS" \
    --feature-mode "$FMODE" \
    --pca-dim "$PCA" \
    --target-abs-max "$TABS" \
    --drop-outlier-flags \
    --calibration bias \
    --aggregate-method median \
    --max-val-mae "$MAXVAL" \
    --max-test-rmse "$MAXTEST" \
    --max-abs-pred "$MAXPRED" \
    --clip-pred-abs "$MAXPRED" \
    "${STACK_ARG[@]}" \
    --addhout-master-csv "$ADDHOUT_MASTER" \
    --addhout-eq-emb-pkl "$ADDHOUT_DUAL_EMB" \
    > "$LOG_DIR/train_graph_ensemble.log" 2>&1

  echo "[DONE] $NAME"
}

for spec in "${EXPERIMENTS[@]}"; do
  run_one "$spec"
done

cat > "$GRID_ROOT/compare_graph_ensemble.py" <<'PY'
import json
from pathlib import Path
import pandas as pd
import numpy as np
root = Path(__file__).resolve().parent
rows = []
for exp in sorted(root.glob('ge*')):
    cfg = {}
    if (exp/'experiment_config.json').exists():
        cfg = json.load(open(exp/'experiment_config.json'))
    m1 = exp/'work'/'test_oof_graph_ensemble_metrics.json'
    m2 = exp/'work'/'addH_out_graph_ensemble_metrics_vs_excel.json'
    runp = exp/'work'/'graph_ensemble_run_summary.csv'
    row = {'exp_name': exp.name, **cfg}
    if m1.exists():
        d = json.load(open(m1)); row.update({f'oof_{k}': v for k,v in d.items()})
    if m2.exists():
        d = json.load(open(m2)); row.update({f'addhout_{k}': v for k,v in d.items()})
    if runp.exists():
        r = pd.read_csv(runp)
        if len(r):
            row['runs_total'] = len(r)
            row['runs_used'] = int(r.get('used', pd.Series(False, index=r.index)).astype(bool).sum())
            row['runs_used_frac'] = row['runs_used']/len(r)
    # selection score: rank-based robust combination
    rows.append(row)
df = pd.DataFrame(rows)
if len(df):
    metrics = []
    for c, asc in [('oof_rmse', True), ('oof_mae', True), ('addhout_mae', True), ('addhout_pearson', False), ('runs_used_frac', False)]:
        if c in df.columns:
            df[f'rank_{c}'] = df[c].rank(ascending=asc, na_option='bottom')
            metrics.append(f'rank_{c}')
    df['selection_score'] = df[metrics].mean(axis=1) if metrics else np.nan
    df = df.sort_values('selection_score')
df.to_csv(root/'graph_ensemble_comparison_summary.csv', index=False)
try:
    df.to_excel(root/'graph_ensemble_comparison_summary.xlsx', index=False)
except Exception:
    pass
print(df.head(30).to_string(index=False))
PY
"$PY_MM" "$GRID_ROOT/compare_graph_ensemble.py" > "$GRID_ROOT/compare.log" 2>&1 || true

echo "[OK] graph ensemble grid finished."
echo "[RESULT] $GRID_ROOT/graph_ensemble_comparison_summary.csv"
echo "[RESULT] $GRID_ROOT/graph_ensemble_comparison_summary.xlsx"
