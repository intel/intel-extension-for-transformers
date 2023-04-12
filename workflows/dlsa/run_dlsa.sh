#!/bin/bash

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

LOG_NAME=$(date "+%m%d-%H%M")
DATASET="sst2"
TRAIN_BATCH_SIZE=1024
EVAL_BATCH_SIZE=109
SEQUENCE_LEN=64
TRAIN_EPOCH=1
MODEL_NAME_OR_PATH="distilbert-base-uncased"
OUTPUT_DIR="./logs"


while [ "$1" != "" ];
do
    case $1 in
    -l | --log_name )
        shift
        LOG_NAME="$1"
        echo "log name is $LOG_NAME"
        ;;
    -d | --dataset )
        shift
        DATASET="$1"
        echo "dataset is : $DATASET"
        ;;
    -s | --sequence_len )
        shift
        SEQUENCE_LEN="$1"
        echo "sequence_len is : $SEQUENCE_LEN"
        ;;
    -o | --output_dir )
        shift
        OUTPUT_DIR="$1"
        echo "OUTPUT_DIR is : $OUTPUT_DIR"
        ;;
    -m | --model )
        shift
        MODEL_NAME_OR_PATH="$1"
        echo "The MODEL_NAME_OR_PATH is : $MODEL_NAME_OR_PATH"
        ;;
    --train_batch_size )
        shift
        TRAIN_BATCH_SIZE="$1"
        echo "batch size for fine-tuning is : $TRAIN_BATCH_SIZE"
        ;;
    --eval_batch_size )
        shift
        EVAL_BATCH_SIZE="$1"
        echo "batch size for inference is: $EVAL_BATCH_SIZE"
        ;;
    -h | --help )
        echo "Usage: bash $0 [OPTIONS]"
        echo "OPTION includes:"
        echo "   -l | --log_name - the log name of this round"
        echo "   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET"
        echo "   -s | --sequence_len - max sequence length"
        echo "   -o | --output_dir - output dir"
        echo "   -m | --model - the input model name or path"
        echo "   --train_batch_size - batch size for fine-tuning"
        echo "   --eval_batch_size - batch size for inference"
        echo "   -h | --help - displays this message"
        exit
      ;;
    * )
        echo "Invalid option: $1"
        echo "Usage: bash $0 [OPTIONS]"
        echo "OPTION includes:"
        echo "   -l | --log_name - the log name of this round"
        echo "   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET"
        echo "   -s | --sequence_len - max sequence length"
        echo "   -o | --output_dir - output dir"
        echo "   -m | --model - the input model name or path"
        echo "   --train_batch_size - batch size for fine-tuning"
        echo "   --eval_batch_size - batch size for inference"
        exit
       ;;
  esac
  shift
done


if [ -z "$LOG_NAME" ]; then
    pre=$(date "+%m%d-%H%M")
else
    pre=$LOG_NAME
fi

OUTPUT_DIR=$OUTPUT_DIR'/'$pre'/'$DATASET
echo "$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"/

export CUDA_VISIBLE_DEVICES="-1"; \
python ./run_dlsa.py \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --do_train \
        --do_quantize \
        --do_predict \
        --max_seq_len "$SEQUENCE_LEN" \
        --num_train_epochs $TRAIN_EPOCH \
        --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
        2>&1 | tee "$OUTPUT_DIR"/test.log
