# Text classification

## GLUE task

This document is used to list steps of introducing [Prune Once For All](https://arxiv.org/abs/2111.05754) examples.

the pattern lock pruning, distillation and quantization aware training are performed simultaneously on the fine tuned model from stage 1 to obtain the quantized model with the same sparsity pattern as the pre-trained sparse language model.

> Note: torch <= 1.9.0+cpu,  4.12.0 <= transformers <= 4.16.0


The following example fine-tunes DistilBERT model of 90% sparsity on the sst-2 task through applying quantization aware-training, pattern lock pruning and distillation simultaneously.

```
python run_glue.py \
    --model_name_or_path Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa \
    --teacher_model distilbert-base-uncased-finetuned-sst-2-english \
    --task_name sst2 \
    --quantization_approach QuantizationAwareTraining \
    --do_train \
    --do_eval \
    --orchestrate_optimization \
    --output_dir ./saved_result  \
    --overwrite_output_dir 
```
