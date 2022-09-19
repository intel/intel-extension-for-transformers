# Text classification

The script `run_glue.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:
### Command

 - To get int8 model

```
python run_glue.py     
    --model_name_or_path bert-base-cased-finetuned-mrpc \
    --task_name mrpc \     
    --tune \     
    --quantization_approach PostTrainingStatic \     
    --do_train \     
    --do_eval \     
    --output_dir ./saved_result \  
    --overwrite_output_dir
```
 - To reload int8 model

```
python run_glue.py     
    --model_name_or_path bert-base-cased-finetuned-mrpc \
    --task_name mrpc \     
    --benchmark \
    --int8 \
    --do_eval \     
    --output_dir ./saved_result \  
    --overwrite_output_dir
```

**Notes**: 
 - Quantization_approach consist of `PostTrainingDynamic`, `PostTrainingStatic`, `QuantizationAwareTraining`.
 - Task name consist of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.


------------------------------------------------------
### Quick start

#### PostTrainingQuantization
 - Topology: 
    - bert_base_mrpc_static

 - To get int8 model

    ```
    cd ptq
    bash run_tuning.sh  --topology=[topology] --output_model=./saved_int8
    ```

 - To reload int8 model

    ```
    cd ptq
    bash run_benchmark.sh --topology=[topology] --config=./saved_int8 --mode=benchmark --int8=true
    ```

#### QuantizationAwareTraining

- Topology: 
    - BERT-MRPC: bert_base_mrpc

 - To get int8 model

    ```
    cd qat
    bash run_tuning.sh  --topology=[topology] --output_model=./saved_int8
    ```

 - To reload int8 model

    ```
    cd qat
    bash run_benchmark.sh --topology=[topology] --config=./saved_int8 --mode=benchmark --int8=true
    ```

### Multi-node usage

We also supported Distributed Data Parallel training on multi nodes settings for quantization.

The default strategy we used is `MultiWorkerMirroredStrategy` in Tensorflow, and with `task_type` set as "worker", we are expected to pass following extra parameters to the script:

* `worker`: a string of your worker ip addresses which is separated by comma and there should not be space between each two of them

* `task_index`: 0 should be set on the chief node (leader) and 1, 2, 3... should be set as the rank of other follower nodes

### Multi-node example

#### To get int8 model

* On leader node

```
bash run_tuning.sh --topology=bert_base_mrpc_static --output_model=./saved_int8 --worker="localhost:12345,localhost:23456"  --task_index=0
```

* On follower node

```
bash run_tuning.sh --topology=bert_base_mrpc_static --output_model=./saved_int8 --worker="localhost:12345,localhost:23456"  --task_index=0
```

Please replace the worker ip address list with your own.

#### To reload int8 model

* On leader node

```
bash run_benchmark.sh --topology=bert_base_mrpc_static --config=./saved_int8 --mode=benchmark --int8=true --worker="localhost:12345,localhost:23456"  --task_index=0
```

* On follower node

```
bash run_benchmark.sh --topology=bert_base_mrpc_static --config=./saved_int8 --mode=benchmark --int8=true --worker="localhost:12345,localhost:23456"  --task_index=0
```

Please replace the worker ip address list with your own.




