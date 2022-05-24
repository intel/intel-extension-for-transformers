#!/bin/bash
# set -x

python run_glue.py \
    --model_name_or_path M-FAC/bert-mini-finetuned-mrpc \
    --task_name mrpc \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./model_and_tokenizer \
    --overwrite_output_dir \
    --to_onnx
