MODEL_NAME_OR_PATH=${MODEL:-meta-llama/Llama-2-13b-hf}
echo "========== using model: ${MODEL_NAME_OR_PATH} =========="
echo "========== START TO MEASURE =========="
QUANT_CONFIG=../quantization_config/maxabs_measure.json python run_streaming_llm.py \
    --model_name_or_path=${MODEL_NAME_OR_PATH} \
    --dataset=emozilla/pg19-test \
    --split=test \
    --attention_sink_window_size=1020 \
    --attention_sink_size=4 \
    --num_sample=1 \
    --num_tokens=2000 \
    --bf16 \
    --use_kv_cache \
    --use_hpu_graphs \
    --perplexity \
    --output_dir=benchmark/fp8_streaming_outputs \
    --overwrite

echo "========== START TO QUANT AND RUN =========="
QUANT_CONFIG=../quantization_config/maxabs_quant.json python run_streaming_llm.py \
    --model_name_or_path=${MODEL_NAME_OR_PATH} \
    --dataset=emozilla/pg19-test \
    --split=test \
    --attention_sink_window_size=1020 \
    --attention_sink_size=4 \
    --num_sample=1 \
    --num_tokens=65000 \
    --bf16 \
    --use_kv_cache \
    --use_hpu_graphs \
    --perplexity \
    --output_dir=benchmark/fp8_streaming_outputs \
    --overwrite \
    --fp8

echo "========== PLOTTING PERPLEXITY =========="
python plot_perplexity.py \
    --features perplexity memory \
    --output_dir benchmark/fp8_streaming_outputs \
    --title "Log perplexity & memory of FP8 model in streaming_llm" \
    --log_perplexity_limit 5.0 \
    --skip_first 100 \
    --figure_dir fp8_streaming_ppl_memory.svg

python plot_perplexity.py \
    --features perplexity latency \
    --output_dir benchmark/fp8_streaming_outputs \
    --title "Log perplexity & latency of FP8 model in streaming_llm" \
    --log_perplexity_limit 5.0 \
    --skip_first 100 \
    --figure_dir fp8_streaming_ppl_latency.svg
