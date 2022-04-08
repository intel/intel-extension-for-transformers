# Question answering

The script `run_qa.py` provides the pruning approach (Magnitude) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Here is how to run the script:
 
```
python run_qa.py \
    --model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --prune \
    --target_sparsity 0.1 \
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