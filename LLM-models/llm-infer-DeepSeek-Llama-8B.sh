# sh examples/custom/infer.sh
CUDA_VISIBLE_DEVICES=1 \
swift infer \
    --adapters llm-output/DeepSeek-R1-Distill-Llama-8B-ALP/v1-20250715-160352/checkpoint-5000 \
    --stream true \
