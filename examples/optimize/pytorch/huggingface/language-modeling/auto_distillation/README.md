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
    --teacher_model_name_or_path bert-large-uncased\
    --data_process_type segment_pair_nsp \
    --max_seq_length 128 \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --output_dir <OUTPUT_DIR>
```