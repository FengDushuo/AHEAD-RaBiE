#!/usr/bin/env bash
set -euo pipefail

PARSED_ROOT="${PARSED_ROOT:-pdf-files-parsed}"
MODEL="${MODEL:-/data/home/terminator/LLM/llm-models/bge-m3}"
QDRANT_PREFIX="${QDRANT_PREFIX:-qdrant_local_shard}"
COLLECTION_PREFIX="${COLLECTION_PREFIX:-papers}"
LOGDIR="${LOGDIR:-logs_qdrant_build}"

BATCH="${BATCH:-16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
FORCE="${FORCE:-0}"

# 只用 1,2,3 三张物理 GPU
GPU_LIST=(${GPU_LIST:-1 2 3})
SHARD_NUM="${SHARD_NUM:-3}"

mkdir -p "$LOGDIR"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

echo "[$(timestamp)] [INFO] Step2 multiview qdrant build starting"
echo "[$(timestamp)] [INFO] Using physical GPUs: ${GPU_LIST[*]}"

if [[ ! -d "$PARSED_ROOT" ]]; then
  echo "[$(timestamp)] [ERROR] Parsed root not found: $PARSED_ROOT"
  exit 1
fi

echo "[$(timestamp)] [INFO] Building entity candidates"
python 2a_build_entity_candidates.py \
  --parsed-root "$PARSED_ROOT" \
  --out "$PARSED_ROOT/entity_candidates.jsonl" \
  > "$LOGDIR/build_entity_candidates.log" 2>&1

if [[ ! -s "$PARSED_ROOT/entity_candidates.jsonl" ]]; then
  echo "[$(timestamp)] [ERROR] entity_candidates.jsonl missing or empty"
  exit 1
fi

run_view_for_all_gpus() {
  local view="$1"
  echo "[$(timestamp)] [INFO] Building view=$view"

  for idx in "${!GPU_LIST[@]}"; do
    local gpu_id="${GPU_LIST[$idx]}"
    local shard_id="$idx"

    CMD=(
      python 2_build_multiview_qdrant.py
      --parsed-root "$PARSED_ROOT"
      --model "$MODEL"
      --device cuda
      --batch "$BATCH"
      --max-length "$MAX_LENGTH"
      --qdrant-path "${QDRANT_PREFIX}${shard_id}"
      --collection-prefix "$COLLECTION_PREFIX"
      --view "$view"
      --shard-id "$shard_id"
      --shard-num "$SHARD_NUM"
    )

    if [[ "$FORCE" == "1" ]]; then
      CMD+=(--force)
    fi

    CUDA_VISIBLE_DEVICES="$gpu_id" "${CMD[@]}" \
      > "$LOGDIR/build_${view}_gpu${gpu_id}_shard${shard_id}.log" 2>&1 &
  done

  wait

  for idx in "${!GPU_LIST[@]}"; do
    local shard_id="$idx"
    STATS_FILE="${QDRANT_PREFIX}${shard_id}/.stats_${COLLECTION_PREFIX}_${view}_shard${shard_id}.json"
    if [[ ! -s "$STATS_FILE" ]]; then
      echo "[$(timestamp)] [ERROR] Missing stats file for view=$view shard=$shard_id : $STATS_FILE"
      exit 1
    fi
  done

  echo "[$(timestamp)] [DONE] view=$view finished"
}

run_view_for_all_gpus "main_text"
run_view_for_all_gpus "table_caption"
run_view_for_all_gpus "entity"

echo "[$(timestamp)] [DONE] multiview qdrant finished"