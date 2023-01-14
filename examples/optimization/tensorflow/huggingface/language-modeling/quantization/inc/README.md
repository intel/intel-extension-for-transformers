# Language Modeling with quantization

There are mainly two kinds of language modeling tasks: Causal Language Modeling (CLM) and Masked Language Modeling (MLM). Two scripts `run_clm.py` and `run_mlm.py` provide quantization examples on the above two kinds of models based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor). Users can easily run the quantization with `run_tuning.sh` and the benchmarking with `run_benchmark.sh`.

Please note that language modeling tasks use `loss` as the evaluation metric so the loss will appear where the accuracy should be in the final tune result statistics, and the `greater_is_better=False` should be set in the Python scripts.

Users can also change the `--max_training_samples`, `--max_eval_samples`, and `--max_seq_length` in the scripts for quicker debugging and to avoid potential lack of memory.

## Command

### CLM

* tuning

```
cd ptq
bash run_tuning.sh  --topology=distilgpt2_clm
```

* evaluating


```
cd ptq
bash run_benchmark.sh --topology=distilgpt2_clm --mode=benchmark --int8=true
```

### MLM

1. distilbert

* tuning

```
cd ptq
bash run_tuning.sh  --topology=distilbert_mlm
```

* evaluating

```
cd ptq
bash run_benchmark.sh --topology=distilbert_mlm --mode=benchmark --int8=true
```

2. distilroberta

* tuning

```
cd ptq
bash run_tuning.sh  --topology=distilroberta_mlm
```

* evaluating

```
cd ptq
bash run_benchmark.sh --topology=distilroberta_mlm --mode=benchmark --int8=true
```