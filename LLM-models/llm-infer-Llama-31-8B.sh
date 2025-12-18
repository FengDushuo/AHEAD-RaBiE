# sh examples/custom/infer.sh
CUDA_VISIBLE_DEVICES=1 \
swift infer \
    --adapters llm-output/Llama-3.1-8B-ALP/v0-20250716-230845/checkpoint-7647 \
    --stream true \
