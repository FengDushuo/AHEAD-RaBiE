# sh examples/custom/infer.sh
CUDA_VISIBLE_DEVICES=1 \
swift app \
    --model checkpoint-10000-merged \
    --stream true \
    --server_port 7860
