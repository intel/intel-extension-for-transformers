python run_streaming_llm.py \
    --model_name_or_path=meta-llama/Llama-2-13b-hf \
    --hf_token=${HF_TOKEN} \
    --dataset_name=emozilla/pg19-test \
    --split=test \
    --trust_remote_code \
    --window_size=1024 \
    --attention_sink_size=4 \
    --num_sample=1 \
    --num_tokens=65000 \
    --enable_streaming \
    --perplexity \
    --output_dir=benchmark/llama2_13b_bf16_outputs \
    --overwrite

python plot_perplexity.py \
    --features perplexity memory \
    --output_dir benchmark/llama2_13b_bf16_outputs \
    --title "Log perplexity & memory of LLAMA2-13B BF16" \
    --log_perplexity_limit 5.0 \
    --skip_first 100 \
    --figure_dir llama2_13b_bf16_ppl.svg
