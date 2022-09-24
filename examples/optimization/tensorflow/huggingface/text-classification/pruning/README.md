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
bash run_benchmark.sh --topology=topology --mode=benchmark --use_pruned_model=true
```
topology is "distilbert_base_sst2"


### Multi-node usage

We also supported Distributed Data Parallel training on multi nodes settings for pruning.

The default strategy we used is `MultiWorkerMirroredStrategy` in Tensorflow, and with `task_type` set as "worker", we are expected to pass following extra parameters to the script:

* `worker`: a string of your worker ip addresses which is separated by comma and there should not be space between each two of them

* `task_index`: 0 should be set on the chief node (leader) and 1, 2, 3... should be set as the rank of other follower nodes

### Multi-node example

* On leader node

```
bash run_tuning.sh --topology=distilbert_base_sst2 --worker="localhost:12345,localhost:23456" --task_index=0
```

which is equal to

```
python run_glue.py \    
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \     
    --task_name sst2 \     
    --prune \      
    --do_train \     
    --do_eval \
    --output_dir ./tmp/sst2_output \  
    --overwrite_output_dir \
    --worker "localhost:12345,localhost:23456" \
    --task_index 0
```

* On follower node

```
bash run_tuning.sh --topology=distilbert_base_sst2 --worker="localhost:12345,localhost:23456" --task_index=1
```

Please replace the worker ip address list with your own.