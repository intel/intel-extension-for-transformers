master_node=$1
rank=$2
torchrun \
            --master_addr=$master_node \
            --nproc_per_node=1 \
            --nnodes=2 \
            --node_rank=$rank \
            ./workflows/chatbot/fine_tuning/instruction_tuning_pipeline/finetune_clm.py \
            --model_name_or_path "mosaicml/mpt-7b" \
            --train_file "/root/chatbot/.github/workflows/sample_data/alpaca_data_sample_45.json" \
            --task completion \
            --dataset_concatenation \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 2 \
            --gradient_accumulation_steps 4 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_total_limit 1 \
            --do_train --learning_rate 1.0e-5 \
            --warmup_ratio 0.03 --weight_decay 0.0 --num_train_epochs 2 --logging_steps 10 \
            --save_steps 2000 --save_total_limit 2 --overwrite_output_dir \
            --output_dir ./mpt-7b-chat_peft_finetuned_model --peft lora --no_cuda --ddp_backend ccl \
	          --trust_remote_code True \
            --tokenizer_name "EleutherAI/gpt-neox-20b" > log${rank}.txt 2>error${rank}.txt &
echo $! > pid${rank}.txt