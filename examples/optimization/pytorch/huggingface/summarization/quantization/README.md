# Summarization

This directory contains examples for finetuning and evaluating transformers on summarization tasks.

`run_summarization.py` is a lightweight example of how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library or use your own files (jsonlines or csv), then fine-tune one of the architectures above on it.

## tune a quantized model with intel_extension_for_transformers

Here is an example on a summarization task:
```bash
python run_summarization.py \
    --model_name_or_path google/pegasus-xsum \
    --dataset_name xsum \
    --do_train \
    --do_eval \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --tune \
    --predict_with_generate
```

T5 model `t5-base` must use an additional argument: `--source_prefix "summarize: "`.

We used CNN/DailyMail dataset in this example as `t5-small` was trained on it and one can get good scores even when pre-training with a very small sample.

Extreme Summarization (XSum) Dataset is another commonly used dataset for the task of summarization. To use it replace `--dataset_name cnn_dailymail --dataset_config "3.0.0"` with  `--dataset_name xsum`.

And here is how you would use it on your own files, after adjusting the values for the arguments
`--train_file`, `--validation_file`, `--text_column` and `--summary_column` to match your setup:

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google/pegasus-xsum \
    --dataset_name xsum \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --tune \
    --predict_with_generate
```
