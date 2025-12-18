# sh examples/custom/infer.sh
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters llm-output/DeepSeek-R1-Distill-Qwen-7B-ALP/v1-20250715-160343/checkpoint-3000 \
    --stream true 
