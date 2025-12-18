# sh examples/custom/infer.sh
CUDA_VISIBLE_DEVICES=0 \
swift app \
    --model llm-output/DeepSeek-R1-Distill-Llama-8B-CES/v0-20250721-213924/checkpoint-10000-merged \
    --stream true \
