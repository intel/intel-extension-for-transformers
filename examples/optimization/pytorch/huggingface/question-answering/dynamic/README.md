# Dynamic-Length Transformer

The implementation is based on [Length Adaptive Transformer](https://github.com/clovaai/length-adaptive-transformer)'s work.
Currently, it supports BERT and RoBERTa based transformers.


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
