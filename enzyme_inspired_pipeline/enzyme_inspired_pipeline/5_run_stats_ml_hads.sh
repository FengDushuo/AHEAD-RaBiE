#!/usr/bin/env bash
set -euo pipefail
DB_OUTDIR="${DB_OUTDIR:-outputs/hads_db}"
OUTROOT="${OUTROOT:-outputs/hads_step5_stats_ml}"
GROUP_COL="${GROUP_COL:-paper_id}"
N_SPLITS="${N_SPLITS:-5}"
SEED="${SEED:-2026}"
QC_FILTER="${QC_FILTER:-1}"
mkdir -p "$OUTROOT/descriptive" "$OUTROOT/predictive" "$OUTROOT/interaction" "$OUTROOT/report"
QC_ARG=()
if [[ "$QC_FILTER" == "1" ]]; then QC_ARG+=(--qc-filter); fi
python 5a_descriptive_stats_hads.py --csv "$DB_OUTDIR/model_feature_table.csv" --outdir "$OUTROOT/descriptive" "${QC_ARG[@]}"
python 5b_predictive_modeling_hads.py --csv "$DB_OUTDIR/model_feature_table.csv" --outdir "$OUTROOT/predictive" --group-col "$GROUP_COL" --n-splits "$N_SPLITS" --seed "$SEED" "${QC_ARG[@]}"
python 5c_interaction_effects_hads.py --csv "$DB_OUTDIR/model_feature_table.csv" --outdir "$OUTROOT/interaction" "${QC_ARG[@]}"
python 5d_interpretability_report_hads.py --model-csv "$DB_OUTDIR/model_feature_table.csv" --descriptive-dir "$OUTROOT/descriptive" --predictive-dir "$OUTROOT/predictive" --interaction-dir "$OUTROOT/interaction" --outdir "$OUTROOT/report"
echo "[DONE] H-ads Step5 -> $OUTROOT"
