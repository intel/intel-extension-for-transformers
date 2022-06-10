# Text classification

## GLUE task

The script `run_glue.py` provides the pruning approach (Magnitude) based on [NLP toolkit].

Here is how to run the script:
 
```
python run_glue.py \    
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \     
    --task_name sst2 \     
    --prune \      
    --do_train \     
    --do_eval \
    --output_dir ./tmp/sst2_output \  
    --overwrite_output_dir
```

### Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```
topology is "distilbert_base_sst2"