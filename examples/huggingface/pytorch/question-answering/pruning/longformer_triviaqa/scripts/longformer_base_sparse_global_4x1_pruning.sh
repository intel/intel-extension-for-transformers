#!/bin/bash
set -x

train_file=./squad-wikipedia-train-4096.json
validation_file=./squad-wikipedia-dev-4096.json
teacher_model=Intel/longformer-base-4096-finetuned-triviaqa

accelerate launch --main_process_port 29745 run_qa_no_trainer.py \
        --model_name_or_path $teacher_model \
        --do_train \
        --do_eval \
        --train_file $train_file \
        --validation_file $validation_file \
        --cache_dir ./tmp_cached \
        --max_seq_length 4096 \
        --doc_stride -1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --per_device_eval_batch_size 1 \
        --num_warmup_steps 1000 \
        --do_prune \
        --target_sparsity 0.8 \
        --pruning_scope "global" \
        --pruning_pattern "4x1" \
        --pruning_frequency 1000 \
        --cooldown_epochs 10 \
        --learning_rate 1e-4 \
        --num_train_epochs 18 \
        --weight_decay  0.01 \
        --output_dir longformer-base-4096-pruned-global-sparse80 \
        --teacher_model_name_or_path $teacher_model \
        --distill_loss_weight 3
