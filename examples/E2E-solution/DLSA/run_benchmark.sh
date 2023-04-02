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
BATCH_SIZE=109
SEQUENCE_LEN=64
MODEL_PATH=""
LOG_DIR="./logs"
DTYPE_INF="fp32"
NUMBER_OF_INSTANCE=2

while [ "$1" != "" ]; do
    case $1 in
    -m | --model_path)
        shift
        MODEL_PATH="$1"
        echo "model path for inference is $MODEL_PATH"
        ;;
    -n | --number_of_instance)
        shift
        NUMBER_OF_INSTANCE="$1"
        echo "The number of instance for inference is $NUMBER_OF_INSTANCE"
        ;;
    -l | --log_name)
        shift
        LOG_NAME="$1"
        echo "log name is $LOG_NAME"
        ;;
    -d | --dataset)
        shift
        DATASET="$1"
        echo "dataset is : $DATASET"
        ;;
    -b | --batch_size)
        shift
        BATCH_SIZE="$1"
        echo "batch size for inference is : $BATCH_SIZE"
        ;;
    -s | --sequence_len)
        shift
        SEQUENCE_LEN="$1"
        echo "sequence_len is : $SEQUENCE_LEN"
        ;;
    --dtype_inf)
        shift
        DTYPE_INF="$1"
        echo "dtype_inf is : $DTYPE_INF"
        ;;
    -h | --help)
        echo "Usage: $0 [OPTIONS]"
        echo "OPTION includes:"
        echo "   -m | --model_path - the model path for inference"
        echo "   -n | --number_of_instance - the number of instance for inference"
        echo "   -l | --log_name - the log name of this round"
        echo "   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET"
        echo "   -b | --batch_size - batch size fine-tuning"
        echo "   -s | --sequence_len - max sequence length"
        echo "   --dtype_inf - [fp32|int8] data type used for inference"
        echo "   -h | --help - displays this message"
        exit
        ;;
    *)
        echo "Invalid option: $1"
        echo "Usage: $0 [OPTIONS]"
        echo "OPTION includes:"
        echo "   -m | --model_path - the model path for inference"
        echo "   -n | --number_of_instance - the number of instance for inference"
        echo "   -l | --log_name - the log name of this round"
        echo "   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET"
        echo "   -b | --batch_size - batch size per instance"
        echo "   -s | --sequence_len - max sequence length"
        echo "   --dtype_inf - [fp32|int8] data type used for inference"
        exit
        ;;
    esac
    shift
done

if [ -z "$MODEL_PATH" ]; then
    echo "Please use -m flag to provide a model for infernece. Use -h flag for more details."
    exit
fi

if [ -z "$LOG_NAME" ]; then
    pre=$(date "+%m%d-%H%M")
else
    pre="$LOG_NAME"_"$DTYPE_INF"
fi

LOG_DIR=$LOG_DIR'/'$pre
echo "The log dir is : ""$LOG_DIR"

mkdir -p "$LOG_DIR"/

export CUDA_VISIBLE_DEVICES="-1"
python ./run_dlsa.py \
        --model_name_or_path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --output_dir "$LOG_DIR" \
        --do_benchmark \
        --dtype_inf "$DTYPE_INF" \
        --max_seq_len "$SEQUENCE_LEN" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --num_of_instance "$NUMBER_OF_INSTANCE" \
        2>&1 | tee "$LOG_DIR"/test.log
