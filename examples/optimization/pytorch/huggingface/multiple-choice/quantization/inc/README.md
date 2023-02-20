Step-by-step
============

This example shows the model quantization for multiple choice task. A multiple choice task is similar to question answering, except several candidate answers are provided along with a context and the model is trained to select the correct answer.

# Prerequisite​

## 1. Environment
```
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Run

The script `run_swag.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

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

# Validated model list

|DATASET|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|SWAG|ehdwns1516/bert-base-uncased_SWAG| ✅| ✅| ✅