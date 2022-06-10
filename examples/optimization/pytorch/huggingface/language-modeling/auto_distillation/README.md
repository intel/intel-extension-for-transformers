# Auto Distillation of Masked Language-Modeling with HuggingFace Transformers
Auto Distillation training script [`run_mlm_autodistillation.py`](./run_mlm_autodistillation.py) is based on [`run_mlm.py`](https://github.com/IntelLabs/Model-Compression-Research-Package/blob/main/examples/transformers/language-modeling/run_mlm.py) of Model-Compression-Research-Package by IntelLabs.

## Data
Datasets are downloaded and processed using `??/datasets` package.

## Usage
The script `run_mlm_autodistillation.py` can be used for auto distillation of `??/transformers` models.

### Auto Distillation
Search best model architectures based on `google/mobilebert-uncased` on English Wikipedia and BookCorpus datasets with `bert-large-uncased` as the teacher model using the following command:

``` bash
python run_mlm_autodistillation.py \
    --config_name mobilebert_config.json \
    --tokenizer_name google/mobilebert-uncased \
    --datasets_name_config wikipedia:20200501.en bookcorpusopen \
    --do_train --do_eval --auto_distillation \
    --teacher_config_name bertlarge_config.json \
    --teacher_model_name_or_path bert-large-uncased \
    --data_process_type segment_pair_nsp \
    --max_seq_length 128 \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --output_dir <OUTPUT_DIR>
```

Search best model architectures based on `prajjwal1/bert-tiny` on English Wikipedia and BookCorpus datasets with `bert-base-uncased` as the teacher model using the following command:

``` bash
python run_mlm_autodistillation.py \
    --config_name bert-tiny_config.json \
    --tokenizer_name prajjwal1/bert-tiny \
    --datasets_name_config wikipedia:20200501.en bookcorpusopen \
    --do_train --do_eval --auto_distillation \
    --teacher_config_name bert-base-uncased_config.json \
    --teacher_model_name_or_path bert-base-uncased \
    --data_process_type segment_pair_nsp \
    --max_seq_length 128 \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --output_dir <OUTPUT_DIR>
```

We also supported Distributed Data Parallel training on multi-GPU in single node setting for Auto Distillation. To use Distributed Data Parallel to speedup training, a machine that has multiple GPUs is needed, the bash command also needs a small adjustment, for example, to search best model architectures based on `google/mobilebert-uncased` through Distributed Data Parallel training, bash command will look like the following, where *<NUM_GPUs>* is the number of GPUs desired to use.

``` bash
python -m torch.distributed.launch --nproc_per_node=<NUM_GPUs> --nnodes=1 --node_rank=0 \
    run_mlm_autodistillation.py \
    --config_name mobilebert_config.json \
    --tokenizer_name google/mobilebert-uncased \
    --datasets_name_config wikipedia:20200501.en bookcorpusopen \
    --do_train --do_eval --auto_distillation \
    --teacher_config_name bertlarge_config.json \
    --teacher_model_name_or_path bert-large-uncased \
    --data_process_type segment_pair_nsp \
    --max_seq_length 128 \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --output_dir <OUTPUT_DIR>
```