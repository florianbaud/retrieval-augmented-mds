#!/bin/bash

export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

# Train
python -u main.py fit --config config.yaml
