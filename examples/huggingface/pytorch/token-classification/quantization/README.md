Step-by-step
============

Token classification assigns a label to individual tokens in a sentence. One of the most common token classification tasks is Named Entity Recognition (NER). NER attempts to find a label for each entity in a sentence, such as a person, location, or organization.

# Prerequisite​

## 1. Environment
```
pip install -r requirements.txt
```


# Run


## 1. Quantization
 
```
 python run_ner.py \
    --model_name_or_path elastic/distilbert-base-uncased-finetuned-conll03-english \
    --dataset_name conll2003 \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --pad_to_max_length \
    --output_dir ./tmp/conll03_output \
    --overwrite_output_dir
```

# Performance Data

|Dataset|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|NER|elastic/distilbert-base-uncased-finetuned-conll03-english| ✅| ✅| ✅

