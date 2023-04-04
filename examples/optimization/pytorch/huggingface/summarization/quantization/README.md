# Step-by-step

This directory contains examples for finetuning and evaluating transformers on summarization tasks.

`run_summarization.py` is a lightweight example of how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library or use your own files (jsonlines or csv), then fine-tune one of the architectures above on it.

# Prerequisite​
## 1. Create Environment
```
pip install intel-intel-for-transformers
pip install -r requirements.txt
# if run pegasus-samsum, need to downgrade the protobuf package to 3.20.x or lower.
pip install protobuf==3.20
```
## Run
## 1. Quantization
For PyTorch, Here is an example on a summarization task:
```bash
python run_summarization.py \
    --model_name_or_path stacked-summaries/flan-t5-large-stacked-samsum-1024 \
    --dataset_name samsum \
    --do_train \
    --do_eval \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --tune \
    --predict_with_generate \
    --perf_tol 0.03
```

T5 model `t5-base` `t5-large` must use an additional argument: `--source_prefix "summarize: "`.

We used CNN/DailyMail dataset in this example as `t5-small` was trained on it and one can get good scores even when pre-training with a very small sample.

Extreme Summarization (XSum) Dataset is another commonly used dataset for the task of summarization. To use it replace `--dataset_name cnn_dailymail --dataset_config "3.0.0"` with  `--dataset_name xsum`.

And here is how you would use it on your own files, after adjusting the values for the arguments
`--train_file`, `--validation_file`, `--text_column` and `--summary_column` to match your setup:

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path stacked-summaries/flan-t5-large-stacked-samsum-1024 \
    --dataset_name samsum \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --tune \
    --predict_with_generate \
    --perf_tol 0.03
```
### 2. Validated Model List
|Dataset|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining 
|---|------------------------------------|---|---|---
|samsum|pegasus_samsum| ✅| N/A | N/A
|cnn_dailymail|t5_base_cnn| ✅| N/A | N/A 
|cnn_dailymail|t5_large_cnn| ✅| N/A| N/A 
|samsum|flan_t5_large_samsum| ✅| ✅| N/A

