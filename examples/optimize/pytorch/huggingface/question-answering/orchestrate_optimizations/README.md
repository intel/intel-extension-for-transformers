# Question answering

This document is used to list steps of introducing [Prune Once For All](https://arxiv.org/abs/2111.05754) examples.

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along the flag --version_2_with_negative.

The following example fine-tunes DistilBERT model of 90% sparsity on the SQuAD1.0 dataset through applying quantization aware-training, pattern lock pruning and distillation simultaneously.
 
```
python run_qa.py \
    --model_name_or_path Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa \
    --teacher_model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --orchestrate_optimizations \
    --do_train \
    --do_eval \
    --output_dir ./tmp/squad_output \
    --overwrite_output_dir
```

### Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```
