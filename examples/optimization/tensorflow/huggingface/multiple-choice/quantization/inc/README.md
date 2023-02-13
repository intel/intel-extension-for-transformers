# Multiple Choice with quantization

The script `run_swag.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

You can use the `run_tuning.sh` and `run_benchmark.sh` to run quantization and evaluation on the `bert-base-uncased-finetuned-swag` model.

## Command

- To get int8 model

```
bash run_tuning.sh  --topology=distilbert_swag
```

- To evaluate with the int8 model


```
bash run_benchmark.sh --topology=distilbert_swag --mode=benchmark --int8=true
```