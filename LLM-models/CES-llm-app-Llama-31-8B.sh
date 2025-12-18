# sh examples/custom/infer.sh
CUDA_VISIBLE_DEVICES=1 \
swift app \
    --model llm-output/Llama-3.1-8B-CES/v0-20250722-142826/checkpoint-10000-merged \
    --stream true \
