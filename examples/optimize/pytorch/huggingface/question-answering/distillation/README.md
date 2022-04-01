# Question answering

The script `run_qa.py` provides the distillation approach based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Here is how to run the script:
 
```
python run_qa.py     
    --model_name_or_path Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa \
    --teacher_model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --distillation \     
    --do_train \ 
    --do_eval \     
    --output_dir ./tmp/squad_output
```

### Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```