# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters llm-output/Llama-3.1-8B-ALP/v0-20250716-230845/checkpoint-7647 \
    --merge_lora true
