python run_streaming_llm.py \
    --model_name_or_path=01-ai/Yi-34B-200K \
    --dataset_name=HuggingFaceH4/mt_bench_prompts \
    --trust_remote_code \
    --window_size=8000 \
    --attention_sink_size=4 \
    --max_new_token=1000 \
    --enable_streaming
