#!/bin/bash
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

python finetuning_gptj.py \
        --model_name_or_path "EleutherAI/gpt-j-6B" \
	    --dataset_name "glue" \
	    --dataset_config_name "mnli" \
	    --config_name ./config.json \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
	    --do_eval \
	    --cache_dir ./gptj_cache \
        --learning_rate 3.434233241010e-4 \
        --num_train_epochs 3 \
        --logging_steps 10 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy steps \
        --bf16 \
        --output_dir ./gptj_finetuned_model >> log.txt


