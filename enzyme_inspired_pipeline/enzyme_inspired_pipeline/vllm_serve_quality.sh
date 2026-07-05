#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/data/home/terminator/LLM/llm-models/glm-4-9b}"
HOST="${HOST:-0.0.0.0}"
LOGDIR="${LOGDIR:-logs}"
PORTS=(8001 8002 8003 8004)
GPUS=(0 1 2 3)

mkdir -p "$LOGDIR"

# Extraction favors deterministic, stable JSON output.
# --served-model-name keeps the OpenAI model id short and consistent.
COMMON_ARGS=(
  --dtype bfloat16
  --enforce-eager
  --trust-remote-code
  --served-model-name hads-extractor
  --disable-log-requests
)

for i in "${!PORTS[@]}"; do
  port="${PORTS[$i]}"
  gpu="${GPUS[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup vllm serve "$MODEL" \
    --host "$HOST" --port "$port" "${COMMON_ARGS[@]}" \
    > "$LOGDIR/vllm_${port}.log" 2>&1 &
  echo "[INFO] launched GPU${gpu} -> port ${port}, log=${LOGDIR}/vllm_${port}.log"
done

echo "[DONE] launched ${#PORTS[@]} vLLM servers"
echo "Use in pipeline:"
echo "  export MODEL_ID=hads-extractor"
echo "  export API_BASES=http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1"
