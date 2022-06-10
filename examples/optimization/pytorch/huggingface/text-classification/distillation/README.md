# Text classification

## GLUE task

The script `run_glue.py` provides the distillation approach based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Here is how to run the script:
 
```
python run_glue.py \
    --model_name_or_path Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa \
    --teacher_model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --task_name sst2 \
    --distillation \
    --do_train \
    --do_eval \
    --output_dir ./tmp/sst2_output
```

### Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```
