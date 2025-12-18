# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters llm-output/DeepSeek-R1-Distill-Qwen-7B-CES/v0-20250721-213918/checkpoint-10000 \
    --merge_lora true
