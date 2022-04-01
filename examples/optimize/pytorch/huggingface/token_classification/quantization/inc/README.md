# Token classification

The script `run_ner.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Here is how to run the script:
 
```
 python run_ner.py     
    --model_name_or_path elastic/distilbert-base-uncased-finetuned-conll03-english \     
    --dataset_name conll2003 \     
    --tune \     
    --quantization_approach PostTrainingStatic \    
    --do_train \     
    --do_eval \     
    --pad_to_max_length \    
    --output_dir ./tmp/conll03_output \ 
    --overwrite_output_dir \
```

### Validated model list

|Task|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|NER|elastic/distilbert-base-uncased-finetuned-conll03-english| ✅| ✅| N/A



### Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```