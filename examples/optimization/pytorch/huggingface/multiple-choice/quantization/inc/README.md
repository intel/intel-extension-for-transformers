# Multiple Choice

The script `run_swag.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Here is how to run the script:

```
python run_swag.py \
    --model_name_or_path ehdwns1516/bert-base-uncased_SWAG \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --pad_to_max_length \
    --output_dir ./tmp/swag_output \
    --overwrite_output_dir
```

### Validated model list

|DATASET|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|SWAG|ehdwns1516/bert-base-uncased_SWAG| ✅| ✅| ✅



### Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```
