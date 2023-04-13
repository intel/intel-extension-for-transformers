Step-by-step
============

This directory contains the example for quantization models on translation tasks. `run_translation.py` is a lightweight examples about how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library or use your own files (jsonlines or csv), then quantize the model on it.

# Prerequisite​

## 1. Environment
```
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Run

## Quantization

Following is an example of a translation MarianMT model to tune a quantized model with Intel Extension for Transformers:

```bash
python examples/pytorch/translation/run_translation.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-ro \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --tune \
    --predict_with_generate
```


T5 models `t5-small` must use an additional argument: `--source_prefix "translate {source_lang} to {target_lang}"`. For example:

```bash
python examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --tune \
    --predict_with_generate
```

If the final BLEU score is far from satisfication, please check out that you didn't forget to use the `--source_prefix` argument.

For the aforementioned group of T5 models it's important to remember that if you switch to a different language pair, make sure to adjust the source and target values in all 3 language-specific command line argument: `--source_lang`, `--target_lang` and `--source_prefix`.
