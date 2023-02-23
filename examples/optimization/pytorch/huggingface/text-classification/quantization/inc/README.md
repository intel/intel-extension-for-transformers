Step-by-Step​
============
The script `run_glue.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

# Prerequisite​
## 1. Create Environment​
Recommend python 3.7 or higher version.
```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Run
## 1. Quantization
GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:
 - To get int8 model

```
python run_glue.py \
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --task_name sst2 \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./saved_result \
    --overwrite_output_dir
```
 - To reload int8 model

```
python run_glue.py \
    --model_name_or_path ./saved_result \
    --task_name sst2 \
    --benchmark \
    --int8 \
    --do_eval \
    --output_dir ./tmp/sst2_output \
    --overwrite_output_dir
```

**Notes**: 
 - Choice of `quantization_approach` can be `PostTrainingDynamic`, `PostTrainingStatic`, and `QuantizationAwareTraining`.
 - Choice of `task_name` can be `cola`, `sst2`, `mrpc`, `stsb`, `qqp`, `mnli`, `qnli`, `rte`, and `wnli`.


## 2. Validated model list

|Task|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|MRPC|textattack/bert-base-uncased-MRPC| ✅| ✅| ✅
|MRPC|textattack/albert-base-v2-MRPC| ✅| ✅| N/A
|SST-2|echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid| ✅| ✅| N/A
|SST-2|distilbert-base-uncased-finetuned-sst-2-english| ✅| ✅| N/A


## 3. Bash Command
### PostTrainingQuantization
 - Topology: 
    - BERT-MRPC: bert_base_mrpc_dynamic, bert_base_mrpc_static
    - BERT-SST2: bert_base_SST-2_dynamic, bert_base_SST-2_static
    - DISTILLBERT-SST2: distillbert_base_SST-2_dynamic, distillbert_base_SST-2_static
    - ALBERT-SST2: albert_base_MRPC_dynamic, albert_base_MRPC_static

 - To get int8 model

    ```
    bash ./ptq/run_tuning.sh  --topology=[topology] --output_model=./saved_int8
    ```

 - To reload int8 model

    ```
    bash ./ptq/run_benchmark.sh --topology=[topology] --config=./saved_int8 --mode=benchmark --int8=true
    ```

### QuantizationAwareTraining

- Topology: 
    - BERT-MRPC: bert_base_mrpc

 - To get int8 model

    ```
    bash ./qat/run_tuning.sh  --topology=[topology] --output_model=./saved_int8
    ```

 - To reload int8 model

    ```
    bash ./qat/run_benchmark.sh --topology=[topology] --config=./saved_int8 --mode=benchmark --int8=true
    ```
