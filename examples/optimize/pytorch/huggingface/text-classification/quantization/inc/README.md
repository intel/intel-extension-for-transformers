# Text classification

The script `run_glue.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:
 
```
python run_glue.py     
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --task_name sst2 \     
    --tune \     
    --quantization_approach PostTrainingStatic \     
    --do_train \     
    --do_eval \     
    --output_dir ./tmp/sst2_output \  
    --overwrite_output_dir

```
where task name can be one of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.

### Validated model list

|Task|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|MRPC|textattack/bert-base-uncased-MRPC| ✅| ✅| ✅
|MRPC|textattack/albert-base-v2-MRPC| ✅| ✅| N/A
|SST-2|echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid| ✅| ✅| N/A
|SST-2|distilbert-base-uncased-finetuned-sst-2-english| ✅| ✅| N/A


### Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```










