accelerate launch run_generation.py \
    --model bigcode/starcoderbase \
    --output_dir "./saved_results" \
    --quantize \
    --int8 \
    --sq \
    --alpha 0.5  \
    --ipex \
    --calib_iters 1 \
    --calib_batch_size 1 \
    --batch_size 1 \
    --accuracy \
    --n_samples 5 \
    --limit 1 \
    --allow_code_execution \
    --do_sample \

