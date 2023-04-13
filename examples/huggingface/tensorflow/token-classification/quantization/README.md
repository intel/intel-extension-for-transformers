Step-by-Step
=========

This document describes the step-by-step instructions for reproducing the quantization on models for the token classification (NER) tasks.

# Prerequisite
## 1. Installation

Make sure you have installed Intel® Extension for Transformers and all the dependencies in the current example:

```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Run

## 1. Run Command (Shell)

- Topology:
   - bert_base_ner

- To get the int8 model

   ```
   cd ptq
   bash run_tuning.sh  --topology=[topology] --output_model=./saved_int8
   ```

- To benchmark the int8 model

   ```
   cd ptq
   bash run_benchmark.sh --topology=[topology] --config=./saved_int8 --mode=benchmark --int8=true
   ```