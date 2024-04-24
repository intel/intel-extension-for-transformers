QUANT_CONFIG=../quantization_config/maxabs_measure.json python run_streaming_llm.py \
    --model_name_or_path=01-ai/Yi-34B-200K \
    --dataset_name=emozilla/pg19-test \
    --split=test \
    --trust_remote_code \
    --window_size=8000 \
    --attention_sink_size=4 \
    --num_sample=1 \
    --num_tokens=64 \
    --enable_streaming \
    --perplexity \
    --output_dir=benchmark/yi_fp8_outputs \
    --overwrite \
    --fp8

QUANT_CONFIG=../quantization_config/maxabs_quant.json python run_streaming_llm.py \
    --model_name_or_path=01-ai/Yi-34B-200K \
    --dataset_name=emozilla/pg19-test \
    --split=test \
    --trust_remote_code \
    --window_size=8000 \
    --attention_sink_size=4 \
    --num_sample=1 \
    --num_tokens=16000 \
    --enable_streaming \
    --perplexity \
    --output_dir=benchmark/yi_fp8_outputs \
    --overwrite \
    --fp8

python plot_perplexity.py \
    --features perplexity memory \
    --output_dir benchmark/yi_fp8_outputs \
    --title "Log perplexity & memory of Yi-38B-200K FP8" \
    --log_perplexity_limit 5.0 \
    --skip_first 100 \
    --figure_dir yi_34b_fp8_ppl.svg
