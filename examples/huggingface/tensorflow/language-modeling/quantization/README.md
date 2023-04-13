Step-by-Step
=========

This document describes the step-by-step instructions for reproducing the quantization on models for the Language Modeling tasks.

There are mainly two kinds of language modeling tasks: Causal Language Modeling (CLM) and Masked Language Modeling (MLM). Two scripts `run_clm.py` and `run_mlm.py` provide quantization examples on the above two kinds of models based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor). Users can easily run the quantization with `run_tuning.sh` and the benchmarking with `run_benchmark.sh`.

Please note that language modeling tasks use `loss` as the evaluation metric so the loss will appear where the accuracy should be in the final tune result statistics, and the `greater_is_better=False` should be set in the Python scripts.

Users can also change the `--max_training_samples`, `--max_eval_samples`, and `--max_seq_length` in the scripts for quicker debugging and to avoid potential lack of memory.

# Prerequisite
## 1. Installation

Make sure you have installed Intel® Extension for Transformers and all the dependencies in the current example:

```shell
pip install intel-extension-for-transformers
cd ptq
pip install -r requirements.txt
```

# Run

## 1. Run Command for the CLM task (Shell)

- Topology:
   - distilgpt2_clm

* To get the int8 model

```
cd ptq
bash run_tuning.sh  --topology=[topology]
```

* To benchmark the int8 model


```
cd ptq
bash run_benchmark.sh --topology=[topology] --mode=benchmark --int8=true
```

## 2. Run Command for the MLM task (Shell)

- Topology:
    - distilbert_mlm
    - distilroberta_mlm

* To get the int8 model

```
cd ptq
bash run_tuning.sh  --topology=[topology]
```

* To benchmark the int8 model

```
cd ptq
bash run_benchmark.sh --topology=[topology] --mode=benchmark --int8=true
```