# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

python finetune_clm.py \
        --model_name_or_path "EleutherAI/gpt-j-6B" \
        --bf16 True \
        --dataset_name  "glue" \
        --dataset_config_name "mnli" \
        --dataset_concatenation \
        --config_name ./config.json \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --do_eval \
        --learning_rate 3.3113761e-4 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./gptj_peft_finetuned_model \
        --peft lora \
        --lora_alpha 54 \
        --lora_target_modules q_proj v_proj k_proj out_proj \
        --use_fast_tokenizer false \
        --use_cpu \
        --task completion \
        --max_train_samples 5000 \
        --max_eval_samples  500
