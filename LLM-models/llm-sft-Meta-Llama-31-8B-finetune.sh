CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --model ./llm-output/Llama-3.1-8B-ALP/v0-20250716-230845/checkpoint-7647-merged \
  --train_type lora \
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
  --output_dir ./llm-output/Llama-3.1-8B-ALP-finetune \
  --attn_impl flash_attn \
  --dataset ./ALP-medline-41112-20250715-alpaca-dataset-finetune.jsonl

