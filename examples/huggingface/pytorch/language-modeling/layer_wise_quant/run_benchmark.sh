#!/bin/bash
set -x

python -u run_clm_no_trainer.py \
    --model /models/opt-125m \
    --output_dir ./saved_results \
    --batch_size 16 \
    --tasks "lambada_openai" \
    --accuracy \
    --int8 