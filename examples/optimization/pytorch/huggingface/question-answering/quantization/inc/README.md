Step-by-Step​
============
The script `run_qa.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

# Prerequisite​
## 1. Create Environment​
Recommend python 3.7 or higher version.
```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Run
## 1. Quantization

For PyTorch, here is how to run the script:

```
python run_qa.py \
    --model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./tmp/squad_output \
    --overwrite_output_dir
```

For IPEX, here is how to run the script:

```
python run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./tmp/squad_output \
    --overwrite_output_dir
    --framework ipex
```
**Note**: support IPEX version > 1.12
## 2. Validated Model List
###  Stock PyTorch Validated model list

|Dataset|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining 
|---|------------------------------------|---|---|---
|squad|distilbert-base-uncased-distilled-squad| ✅| ✅| ✅

###  Intel Extension for PyTorch (IPEX) Validated model list
|Dataset|Pretrained model|Supported IPEX Version 
|---|------------------------------------|---
|squad|distilbert-base-uncased-distilled-squad| == 1.13
|squad|bert-large| >= 1.12