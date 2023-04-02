#!/bin/bash
set -x

train_file=./squad-wikipedia-train-4096.json
validation_file=./squad-wikipedia-dev-4096.json
pretrained_model=allenai/longformer-base-4096

accelerate launch --main_process_port 29245 run_qa_no_trainer.py \
    --model_name_or_path $pretrained_model \
    --do_train \
    --do_eval \
    --train_file $train_file \
    --validation_file $validation_file \
    --cache_dir ./tmp_cached \
    --max_seq_length 4096 \
    --doc_stride -1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --num_warmup_steps 1000 \
    --learning_rate 3.5e-5 \
    --num_train_epochs 4 \
    --output_dir longformer-base-4096-dense-baseline
