# Token classification with quantization

The script `run_ner.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

You can use the `run_tuning.sh` and `run_benchmark.sh` to run quantization and evaluation on the `bert_base_ner` model.

## Command

 - To get int8 model

```
bash run_tuning.sh  --topology=bert_base_ner
```

 - To evaluate with the int8 model

```
bash run_benchmark.sh --topology=bert_base_ner --mode=benchmark --int8=true
```