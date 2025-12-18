# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters llm-output/DeepSeek-R1-Distill-Qwen-7B-ALP/v1-20250715-160343/checkpoint-7647 \
    --merge_lora true
