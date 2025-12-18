# sh examples/custom/infer.sh
CUDA_VISIBLE_DEVICES=1 \
swift infer \
    --model llm-output/Llama-3.1-8B-ALP-finetune/v0-20250718-001606/checkpoint-10000-merged \
    --stream true \
