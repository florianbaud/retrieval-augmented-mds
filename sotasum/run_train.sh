#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 python main.py \
    --deepspeed-log \
    --use-cache \
    --use-own-decoder \
    --gates-mode nmt \
    --skip-residual \
    --copy-decoder-layers 1 \
    --copy-forcing 0.5 \
    --disable-mlflow \
    --gradient-checkpoint \
    --data-workers 0 \
    --mips-freezed \
    --mips-num-gpus 1 \
    --mips-db-max-size 64 \
    --mips-batch-size 32 \
    --mips-rebuild-every 650 \
    --mips-string-factory Flat \
    --mips-metric-type 1 \
    --mips-train-size 3000 \
    --mips-topk 5 \
    --mips-tok-max-length 512 \
    --memory-tok-max-length 512 \
    --query-tok-max-length 512 \
    --model-tok-max-length 4096 \
    --mips-model-name allenai/longformer-large-4096 \
    --query-encoder-path allenai/longformer-large-4096 \
    --pb-refresh-rate 1 \
    --accumulate_grad_batches 2 \
    --warmup-steps 2000 \
    --label-smoothing-eps 0.1 \
    --num-beams 3 \
    --lr 1e-4 \
    --max_epochs 10 \
    --data-path ../data_hf \
    --checkpoint-dir ../checkpoint \
    --accelerator gpu \
    --devices 2 \
    --precision 16 \
    --gradient_clip_val 1.0 \
    --batch-size 1 \
    --check_val_every_n_epoch 1 \
    --val_check_interval 1.0 \
    --validation-batch-size 8 \
    --validation-outputs-dir ../outputs \
    --num_sanity_val_steps 0 \
    --model-cache-dir ../pretrainedmodels \
    --model-name allenai/led-large-16384
