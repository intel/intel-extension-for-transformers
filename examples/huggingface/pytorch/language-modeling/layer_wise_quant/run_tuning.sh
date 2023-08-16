#!/bin/bash
python -u run_clm_no_trainer.py \
    --model /models/opt-125m \
    --output_dir ./saved_results \
    --dataset /dataset/pile_10k \
    --quantize \
    --batch_size 112 \
    --layer_wise \
    --accuracy \
    --int8 \
    --ipex \
    --sq \
    --alpha 0.5