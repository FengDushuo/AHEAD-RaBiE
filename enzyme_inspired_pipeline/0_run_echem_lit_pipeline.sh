#!/usr/bin/env bash
set -euo pipefail
# Full H adsorption / deprotonation literature-mining pipeline.
# Set RUN_STEP1=0 or RUN_STEP2=0 when parsed PDFs / Qdrant shards already exist.
RUN_STEP1="${RUN_STEP1:-1}"
RUN_STEP2="${RUN_STEP2:-1}"
RUN_STEP3="${RUN_STEP3:-1}"
RUN_STEP4="${RUN_STEP4:-1}"
RUN_STEP5="${RUN_STEP5:-1}"

if [[ "$RUN_STEP1" == "1" ]]; then bash 1_parse_echem_corpus.sh; fi
if [[ "$RUN_STEP2" == "1" ]]; then bash 2_build_multiview_qdrant_4gpu.sh; fi
if [[ "$RUN_STEP3" == "1" ]]; then bash 3_run_extraction_pipeline_hads.sh; fi
if [[ "$RUN_STEP4" == "1" ]]; then bash 4_build_hads_normalized_database.sh; fi
if [[ "$RUN_STEP5" == "1" ]]; then bash 5_run_stats_ml_hads.sh; fi
