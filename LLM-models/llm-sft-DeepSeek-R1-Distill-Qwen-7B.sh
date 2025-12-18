# sh examples/custom/sft.sh
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model ./llm-models/DeepSeek-R1-Distill-Qwen-7B \
    --train_type lora \
    --dataset ./ALP-medline-41112-20250715-alpaca-dataset.jsonl \
    --model_type deepseek_r1_distill \
    --output_dir ./llm-output/DeepSeek-R1-Distill-Qwen-7B-ALP \
    --lora_rank 8 \
    --lora_alpha 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 5 \
    --max_length 2048 \
    --max_steps 100000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --gradient_accumulation_steps 4 \
    --attn_impl flash_attn 
