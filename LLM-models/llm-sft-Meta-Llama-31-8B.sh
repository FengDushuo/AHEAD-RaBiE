# sh examples/custom/sft.sh
NPROC_PER_NODE=2
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model ./llm-models/Llama-3.1-8B \
    --train_type lora \
    --dataset ./ALP-medline-41112-20250715-alpaca-dataset.jsonl \
    --model_type llama \
    --output_dir ./llm-output/Llama-3.1-8B-ALP \
