# YAML Config

Distillation
```
model_name_or_path:         Path to pretrained model or model identifier from huggingface.co/models.
teacher_model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models which will be teacher model
task_name:                  The name of the task to train on.
distillation:               Needs to be true in this case.
do_train:                   Whether to do training.
do_eval:                    Whether to do evaluation.
max_train_samples:          Maximum number of training samples.
output_dir:                 Path to output directory.
overwrite_output_dir:       Whether to overwrite Output cache.
perf_tol:                   Performance tolerance when optimizing the model.
```

Quantization
```
model_name_or_path:         Path to pretrained model or model identifier from huggingface.co/models.
task_name:                  The name of the task to train on.
do_eval:                    Whether to do evaluation.
output_dir:                 Path to output directory.
overwrite_output_dir:       Whether to overwrite Output cache.
perf_tol:                   Performance tolerance when optimizing the model.
quantization:               Needs to be true in this case.
quantization_approach:      Quantization approach. Supported approach are PostTrainingStatic, PostTrainingDynamic and QuantizationAwareTraining.
is_relative:                Metric tolerance model, expected to be relative or absolute.
int8:                       Load int8 model.
```

Distillation + Quantization
```
model_name_or_path:         Path to pretrained model or model identifier from huggingface.co/models.
teacher_model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models which will be teacher model
task_name:                  The name of the task to train on.
distillation:               Needs to be true in this case.
do_train:                   Whether to do training.
do_eval:                    Whether to do evaluation.
max_train_samples:          Maximum number of training samples.
output_dir:                 Path to output directory.
overwrite_output_dir:       Whether to overwrite Output cache.
perf_tol:                   Performance tolerance when optimizing the model.
quantization:               Needs to be true in this case.
quantization_approach:      Quantization approach. Supported approach are PostTrainingStatic, PostTrainingDynamic and QuantizationAwareTraining.
is_relative:                Metric tolerance model, expected to be relative or absolute.
int8:                       Load int8 model.
```
