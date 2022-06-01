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
    bash ./ptq/run_tuning.sh  --topology=[topology] --output_model=./saved_int8
    ```

 - To reload int8 model

    ```
    bash ./ptq/run_benchmark.sh --topology=[topology] --config=./saved_int8 --mode=benchmark --int8=true
    ```

#### QuantizationAwareTraining

- Topology: 
    - BERT-MRPC: bert_base_mrpc

 - To get int8 model

    ```
    bash ./qat/run_tuning.sh  --topology=[topology] --output_model=./saved_int8
    ```

 - To reload int8 model

    ```
    bash ./qat/run_benchmark.sh --topology=[topology] --config=./saved_int8 --mode=benchmark --int8=true
    ```









