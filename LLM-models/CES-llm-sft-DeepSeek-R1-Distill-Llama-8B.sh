# sh examples/custom/sft.sh
CUDA_VISIBLE_DEVICES=1 \
swift sft \
    --model ./llm-models/DeepSeek-R1-Distill-Llama-8B \
    --train_type lora \
    --dataset ./CES-medline-1437-20250721-alpaca-dataset.jsonl \
    --model_type deepseek_r1_distill \
    --output_dir ./llm-output/DeepSeek-R1-Distill-Llama-8B-CES \
    --lora_rank 8 \
    --lora_alpha 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 5 \
    --max_length 2048 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --gradient_accumulation_steps 4 \
    --attn_impl flash_attn 
