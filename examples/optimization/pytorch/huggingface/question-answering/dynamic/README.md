# Quantized Length Adaptive Transformer

The implementation is based on [Length Adaptive Transformer](https://github.com/clovaai/length-adaptive-transformer)'s work.
Currently, it supports BERT and RoBERTa based transformers.

[QuaLA-MiniLM: A Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114) has been accepted by NeurIPS 2022. Our quantized length-adaptive MiniLM model (QuaLA-MiniLM) is trained only once, dynamically fits any inference scenario, and achieves an accuracy-efficiency trade-off superior to any other efficient approaches per any computational budget on the SQuAD1.1 dataset (up to x8.8 speedup with <1% accuracy loss). The following shows how to reproduce this work and we also provide the [jupyter notebook tutorials](../../../../../../docs/tutorials/pytorch/question-answering/Dynamic_MiniLM_SQuAD.ipynb).

## Training


### Step 1: Finetuning Pretrained Transformer
```
python run_qa.py \
--model_name_or_path bert-base-uncased \
--dataset_name squad \
--do_train \
--do_eval \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--per_device_train_batch_size 8 \
--output_dir output/finetuning
```


### Step 2: Training with LengthDrop

```
python run_qa.py \
--model_name_or_path output/finetuning \
--dataset_name squad \
--do_train \
--do_eval \
--learning_rate 3e-5 \
--num_train_epochs 5 \
--max_seq_length 384 \
--doc_stride 128 \
--per_device_train_batch_size 8 \
--length_adaptive \
--num_sandwich 2  \
--length_drop_ratio_bound 0.2 \
--layer_dropout_prob 0.2 \
--output_dir output/dynamic 

```

### Step 3: Evolutionary Search

run search to optimize length configurations for any possible target computational budget.

```
python run_qa.py \
--model_name_or_path output/dynamic \
--dataset_name squad \
--max_seq_length 384 \
--doc_stride 128 \
--do_eval \
--per_device_eval_batch_size 32 \
--do_search \
--output_dir output/search

```


### Step 4: Quantization

```
python run_qa.py \
--model_name_or_path "sguskin/dynamic-minilmv2-L6-H384-squad1.1" \
--dataset_name squad \
--quantization_approach PostTrainingStatic \
--do_eval \
--do_train \
--tune \
--output_dir output/quantized-dynamic-minilmv \
--overwrite_cache \
--per_device_eval_batch_size 32
```


### Step 5: Apply Length Config for Quantization
```
python run_qa.py \
--model_name_or_path output/quantized-dynamic-minilmv \
--dataset_name squad \
--do_eval \
--accuracy_only \
--int8 \
--output_dir output/quantized-dynamic-minilmv \
--overwrite_cache \
--per_device_eval_batch_size 32 \
--length_config "(315, 251, 242, 159, 142, 33)"
```
