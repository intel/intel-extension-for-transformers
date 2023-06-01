# YAML Config

Fine Tuning
```
model_name_or_path :             Path to pretrained model or model identifier from huggingface.co/models.
tokenizer_name:                  Pretrained tokenizer name or path if not the same as model_name.
dataset:                         Local or Huggingface datasets name.

""" Required only when dataset: 'local' """
local_dataset:
    finetune_input :             Input filename incase of local dataset.
    delimiter:                   File delimiter.
    features:
        class_label:             Label column name.
        data_column:             Data column name.
        id:                      Id column name.
    label_list:                  List of class labels.

pipeline:                        The pipeline to use. 'finetune' in this case.
finetune_impl:                   The implementation of fine-tuning pipeline. Now we support trainer and itrex implementation.
dtype_ft:                        Data type for finetune pipeline. Support fp32 and bf16 for CPU. Support fp32, tf32, and fp16 for GPU.
max_seq_len:                     The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
smoke_test:                      Whether to execute in sanity check mode.
max_train_samples:               For debugging purposes or quicker training, truncate the number of training examples to this value if set.
max_test_samples:                For debugging purposes or quicker testing, truncate the number of testing examples to this value if set.
preprocessing_num_workers:       The number of processes to use for the preprocessing.
overwrite_cache:                 Overwrite the cached training and evaluation sets.
finetune_output:                 Path of file to write output results.

training_args:
    num_train_epochs:            Number of epochs to run.
    do_train:                    Whether to run training.
    do_predict:                  Whether to run predictions.
    per_device_train_batch_size: Batch size per device during training.
    per_device_eval_batch_size:  Batch size per device during evaluation.
    output_dir:                  Output directory.
```

Inference
```
model_name_or_path :             Path to pretrained model or model identifier from huggingface.co/models.
tokenizer_name:                  Pretrained tokenizer name or path if not the same as model_name.
dataset:                         Local or Huggingface datasets name.

""" Required only when dataset: 'local' """
local_dataset:
    inference_input :            Input filename incase of local dataset.
    delimiter:                   File delimiter.
    features:
        class_label:             Label column name.
        data_column:             Data column name.
        id:                      Id column name.
    label_list:                  List of class labels.

pipeline:                        The pipeline to use. 'inference' in this case.
infer_impl:                      The implementation of inference pipeline. Now we support trainer and itrex implementation.
dtype_inf:                       Data type for inference pipeline. Support fp32 and bf16 for CPU. Support fp32, tf32, and fp16 for GPU.
max_seq_len:                     The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
smoke_test:                      Whether to execute in sanity check mode.
max_train_samples:               For debugging purposes or quicker training, truncate the number of training examples to this value if set.
max_test_samples:                For debugging purposes or quicker testing, truncate the number of testing examples to this value if set.
preprocessing_num_workers:       The number of processes to use for the preprocessing.
overwrite_cache:                 Overwrite the cached training and evaluation sets.
inference_output:                Path of file to write output results.
multi_instance:                  Whether to use multi-instance mode.

training_args:
    num_train_epochs:            Number of epochs to run.
    do_train:                    Whether to run training.
    do_predict:                  Whether to run predictions.
    per_device_train_batch_size: Batch size per device during training.
    per_device_eval_batch_size:  Batch size per device during evaluation.
    output_dir:                  Output directory.
```

Inference Only
```
model_name_or_path :             Path to pretrained model or model identifier from huggingface.co/models.
tokenizer_name:                  Pretrained tokenizer name or path if not the same as model_name.
dataset:                         Local or Huggingface datasets name. (Needs to be 'local' in this case)

""" Required only when dataset: 'local' """
local_dataset:
    inference_input :            List of input filenames incase of local dataset.
    delimiter:                   File delimiter.
    features:
        class_label:             Label column name.
        data_column:             Data column name.
        id:                      Id column name.
    label_list:                  List of class labels.

pipeline:                        The pipeline to use. 'inference_only' in this case.
infer_impl:                      The implementation of inference pipeline. Now we support trainer and itrex implementation.
dtype_inf:                       Data type for inference pipeline. Support fp32 and bf16 for CPU. Support fp32, tf32, and fp16 for GPU.
max_seq_len:                     The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
smoke_test:                      Whether to execute in sanity check mode.
max_train_samples:               For debugging purposes or quicker training, truncate the number of training examples to this value if set.
max_test_samples:                For debugging purposes or quicker testing, truncate the number of testing examples to this value if set.
preprocessing_num_workers:       The number of processes to use for the preprocessing.
overwrite_cache:                 Overwrite the cached training and evaluation sets.
inference_output:                Path of file to write output results.
multi_instance:                  Whether to use multi-instance mode.

training_args:
    num_train_epochs:            Number of epochs to run.
    do_train:                    Whether to run training.
    do_predict:                  Whether to run predictions.
    per_device_train_batch_size: Batch size per device during training.
    per_device_eval_batch_size:  Batch size per device during evaluation.
    output_dir:                  Output directory.
```
