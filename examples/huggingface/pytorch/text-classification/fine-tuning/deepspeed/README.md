<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Text classification examples with Deepspeed

## Deepspeed integration

This example shows integration Huggingface scripts with Deepspeed doing fine-tuning tasks

Here is some tested features:

* bf16 precision
* ZeRO stage 0/1/2/3
* ZeRO Offload(optimizer/param)
* activation checkpointing
* LoRA

## GLUE tasks

Based on the huggingface script [`run_glue_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py).

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/). This script can fine-tune any of the models on the [hub](https://huggingface.co/models)
and can also be used for a dataset hosted on our [hub](https://huggingface.co/datasets) or your own data in a csv or a JSON file
(the script might need some tweaks in that case, refer to the comments inside for help).

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:

```bash
export TASK_NAME=mrpc

deepspeed --num_gpus=12 run_glue_deepspeed.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir log/Llama/$TASK_NAME/
```
where task name can be one of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.





