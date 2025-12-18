# sh examples/custom/infer.sh
CUDA_VISIBLE_DEVICES=0 \
swift app \
    --model llm-output/DeepSeek-R1-Distill-Qwen-7B-CES/v0-20250721-213918/checkpoint-10000-merged \
    --stream true \
