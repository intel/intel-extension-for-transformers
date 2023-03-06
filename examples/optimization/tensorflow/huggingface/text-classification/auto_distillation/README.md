Step-by-Step
=========

This document describes the step-by-step instructions for reproducing the auto distillation on models for the text classification (GLUE) tasks.

# Prerequisite
## 1. Installation

Make sure you have installed Intel® Extension for Transformers and all the dependencies in the current example:

```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Run

## 1. Run Command (Shell)

- Topology:
   - distilbert-base-uncased

```
bash run_tuning.sh  --topology=[topology]
```

```
bash run_benchmark.sh --topology=[topology] --mode=benchmark --use_distillation_model=true
```


## 2. Run Command (Python)

```
python -u ./run_glue.py \
        --model_name_or_path distilbert-base-uncased \
        --teacher_model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
        --task_name sst2 \
        --temperature 1.0 \
        --autodistill \
        --loss_types CE CE \
        --layer_mappings classifier classifier \
        --do_eval \
        --do_train \
        --output_dir ./tmp/sst2_output \
        --overwrite_output_dir \
```

# Multi-node Mode

The script `run_glue.py` also provides multinode mode. Just need add three extra arguments to enable multinode.

Here is how to run the script on one worker:
```
python -u ./run_glue.py \
        --model_name_or_path ${model_name_or_path} \
        --teacher_model_name_or_path ${teacher_model_name_or_path} \
        --task_name ${TASK_NAME} \
        --temperature 1.0 \
        --autodistill \
        --loss_types CE CE \
        --layer_mappings classifier classifier \
        --do_eval \
        --do_train \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --overwrite_output_dir \
        --overwrite_cache \
        --multinode \
        --worker "woker0.ipaddress:port,woker1.ipaddress:port" \
        --task_index 0 \
```
To rank other follower nodes, change the `task_index` to 1, 2, 3 and etc.

Or you can run the shell script instead:
```
bash run_tuning_multinode.sh  --topology=distilbert-base-uncased --worker=ip_address_list --task_index=0
```