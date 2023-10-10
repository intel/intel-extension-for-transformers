cd /root/chatbot
hname=$(hostname -s)
python3 workflows/chatbot/fine_tuning/instruction_tuning_pipeline/finetune_clm.py \
    --model_name_or_path mosaicml/mpt-7b-chat \
    --train_file .github/workflows/sample_data/alpaca_data_sample_45.json \
    --bf16 False \
    --output_dir ./mpt_peft_finetuned_model \
    --num_train_epochs 1 \
    --max_steps 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4  \
    --logging_steps 1 \
    --peft lora \
    --group_by_length True \
    --dataset_concatenation \
    --do_train \
    --trust_remote_code True \
    --tokenizer_name "EleutherAI/gpt-neox-20b" \
    --use_fast_tokenizer True \
    --max_eval_samples 64 \
    --no_cuda \
    --ddp_backend ccl >"${hname}.log" 2>"${hname}.err"
