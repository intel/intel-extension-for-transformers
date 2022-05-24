#!/bin/bash
# set -x

python run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./model_and_tokenizer \
    --overwrite_output_dir \
    --to_onnx
