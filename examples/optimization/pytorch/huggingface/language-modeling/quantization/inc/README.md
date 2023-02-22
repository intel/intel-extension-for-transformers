Step-by-Step​
============
The scripts `run_clm.py`, `run_mlm.py` and `run_plm.py` provide three quantization approaches respectively (PostTrainingDynamic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

# Prerequisite​
## 1. Create Environment​
Recommend python 3.7 or higher version.
```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Run
## 1. Quantization
Here is how to run the scripts:

**Causal Language Modeling (CLM)**

```
python run_clm.py \
    --model_name_or_path EleutherAI/gpt-neo-125M \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./tmp/clm_output \
    --overwrite_output_dir

```

**Masked Language Modeling (MLM)**

```
python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./tmp/mlm_output \
    --overwrite_output_dir
```

**Permutation Language Modeling (PLM)**

```
    python run_plm.py \
    --model_name_or_path xlnet-base-cased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./tmp/plm_output \
    --overwrite_output_dir

```

## 2. Validated Model List

|Type|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|CLM|EleutherAI/gpt-neo-125M| ✅| ✅| ✅
|CLM|EleutherAI/gpt-j-6B| ✅| ✅| Stay tuning
|MLM|bert-base-uncased| ✅| ✅| ✅
|PLM|xlnet-base-cased| ✅| ✅| ✅

## 3. Bash Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```
> NOTE
>
> topology should be one of: {"gpt_neo_clm_static", "gpt_neo_clm_dynamic", "gptj_clm_static", "gptj_clm_dynamic", "bert_mlm_static", "bert_mlm_dynamic", "xlnet_plm_static", "xlnet_plm_dynamic", "reformer_crime_and_punishment_static", "ctrl_wikitext_static"}