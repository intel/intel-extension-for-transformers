MODEL_NAME_OR_PATH=${MODEL:-01-ai/Yi-34B-Chat}
echo "========== using model: ${MODEL_NAME_OR_PATH} =========="
python run_streaming_llm.py \
    --model_name_or_path=${MODEL_NAME_OR_PATH} \
    --dataset=HuggingFaceH4/mt_bench_prompts \
    --attention_sink_window_size=1020 \
    --attention_sink_size=4 \
    --max_new_token=512 \
    --num_sample=-1 \
    --bf16 \
    --use_kv_cache \
    --use_hpu_graphs
