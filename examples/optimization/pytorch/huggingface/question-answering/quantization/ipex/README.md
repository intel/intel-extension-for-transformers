# Question answering
The script `run_qa.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Here is how to run the script:

```
python run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-maskinuned-squad \
    --dataset_name squad \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./tmp/squad_output \
    --overwrite_output_dir
```
### Validated model list

|Dataset|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|squad|distilbert-base-uncased-distilled-squad| N/A| ✅| N/A
|squad|bert-large-uncased-whole-word-maskinuned-squad| N/A| ✅| N/A
