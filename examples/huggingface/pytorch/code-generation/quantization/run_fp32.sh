accelerate launch run_generation.py \
    --model bigcode/starcoderbase \
    --output_dir "./saved_results" \
    --ipex \
    --batch_size 10 \
    --accuracy \
    --n_samples 10 \
    --allow_code_execution \
    --do_sample \