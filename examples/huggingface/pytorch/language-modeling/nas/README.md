Step-by-Step​
============

Neural architecture search (NAS) training script [`run_mlm.py`](./run_mlm.py) is based on [`run_mlm.py`](https://github.com/IntelLabs/Model-Compression-Research-Package/blob/main/examples/transformers/language-modeling/run_mlm.py) of Model-Compression-Research-Package by IntelLabs.

# Prerequisite​

## 1. Environment​
Recommend python 3.7 or higher version.
```shell
pip install -r requirements.txt
```

## 2. Prepare Dataset
Datasets are downloaded and processed using `🤗 Datasets` package.

# NAS

## 1. Usage
The script `run_mlm.py` can be used for NAS of `🤗 Transformers` models.

### 1.1 MobileBERT
Search best model architectures based on `google/mobilebert-uncased` on English Wikipedia and BookCorpus datasets using the following command:

``` bash
python run_mlm.py \
    --model_name_or_path google/mobilebert-uncased \
    --datasets_name_config wikipedia:20200501.en bookcorpusopen \
    --do_train --do_eval --nas \
    --data_process_type segment_pair_nsp \
    --max_seq_length 128 \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --output_dir <OUTPUT_DIR>
```

### 1.2 BERTTiny
Search best model architectures based on `prajjwal1/bert-tiny` on English Wikipedia and BookCorpus datasets using the following command:

``` bash
python run_mlm.py \
    --model_name_or_path prajjwal1/bert-tiny \
    --datasets_name_config wikipedia:20200501.en bookcorpusopen \
    --do_train --do_eval --nas \
    --data_process_type segment_pair_nsp \
    --max_seq_length 128 \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --output_dir <OUTPUT_DIR>
```
