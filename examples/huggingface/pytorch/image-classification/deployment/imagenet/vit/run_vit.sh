#!/bin/bash
# set -x
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

OUTPUT_DIR=$(pwd)
LOG_NAME="vit.log"
DATASET="imagenet-1k"
MODEL_NAME_OR_PATH="google/vit-base-patch16-224"
BATCH_SIZE=1
WARM_UP=5
SEQUENCE_LEN=1
ITERATION=10
PRECISION="int8"
CACHE_DIR="./tmp"
MODE="performance"

for var in "$@"
do
    case $1 in
    --log_name=*)
        LOG_NAME=$(echo $var |cut -f2 -d=)
        echo "log name prefix is $LOG_NAME"
        ;;
    --output=*)
        OUTPUT_DIR=$(echo $var |cut -f2 -d=)
        echo "output location is $OUTPUT_DIR"
        ;;
    --dataset=*)
        DATASET=$(echo $var |cut -f2 -d=)
        echo "dataset is : $DATASET"
        ;;
    --cache_dir=*)
        CACHE_DIR=$(echo $var |cut -f2 -d=)
        echo "cache location is : $CACHE_DIR"
        ;;
    --sequence_len=*)
        SEQUENCE_LEN=$(echo $var |cut -f2 -d=)
        echo "sequence_len is : $SEQUENCE_LEN"
        ;;
    --model=*)
        MODEL_NAME_OR_PATH=$(echo $var |cut -f2 -d=)
        echo "The MODEL_NAME_OR_PATH is : $MODEL_NAME_OR_PATH"
        ;;
    --precision=*)
        PRECISION=$(echo $var |cut -f2 -d=)
        echo "The PRECISION is : $PRECISION"
        ;;
    --batch_size=*)
        BATCH_SIZE=$(echo $var |cut -f2 -d=)
        echo "batch size for inference is: $BATCH_SIZE"
        ;;
    --warm_up=*)
        WARM_UP=$(echo $var |cut -f2 -d=)
        echo "warm up for inference is: $WARM_UP"
        ;;
    --iteration=*)
        ITERATION=$(echo $var |cut -f2 -d=)
        echo "iteration for inference is: $ITERATION"
        ;;
    --mode=*)
        MODE=$(echo $var |cut -f2 -d=)
        echo "inference mode is: $MODE"
        ;;
    -h | --help )
        echo "Usage: bash $0 [OPTIONS]"
        echo "OPTION includes:"
        echo "   --LOG_NAME - the log name prefix"
        echo "   --OUTPUT_DIR - the output location"
        echo "   --dataset - dataset for model optimization"
        echo "   --sequence_len - max sequence length"
        echo "   --cache_dir - use cache to speed up"
        echo "   --model - the input model name or path"
        echo "   --precision - model precision for inference"
        echo "   --mode - inference mode, choose from accuracy or throughput or latency"
        echo "   --batch_size - batch size for inference"
        echo "   --iteraion - iteration for inference"
        echo "   --warm_up - warm up steps for inference"
        echo "   --help - displays this message"
        exit
      ;;
    * )
        echo "Invalid option: $1"
        echo "Usage: bash $0 [OPTIONS]"
        echo "OPTION includes:"
        echo "   --OUTPUT_DIR - the output location"
        echo "   --dataset - dataset for model optimization"
        echo "   --sequence_len - max sequence length"
        echo "   --cache_dir - use cache to speed up"
        echo "   --model - the input model name or path"
        echo "   --precision - model precision for inference"
        echo "   --mode - inference mode, choose from accuracy or throughput or latency"
        echo "   --batch_size - batch size for inference"
        echo "   --iteraion - iteration for inference"
        echo "   --warm_up - warm up steps for inference"
        exit
       ;;
  esac
  shift
done
## 
inference_model="./model_and_tokenizer/${PRECISION}-model.onnx"
if [[ ${PRECISION} = 'dynamic_int8' ]]; then
    inference_model="./model_and_tokenizer/fp32-model.onnx"
fi
if [[ -f ${inference_model} ]]; then
    echo "=====   Load ONNX Model ${MODEL_NAME_OR_PATH} from local ======"
else
    echo "==========    Prepare Model ${MODEL_NAME_OR_PATH} with Precision ${PRECISION} ========"
    mode_cmd=""
    if [[ ${PRECISION} = 'int8' ]]; then
        mode_cmd=$mode_cmd" --tune --quantization_approach PostTrainingStatic"
    elif [[ ${PRECISION} = 'bf16' ]]; then
        mode_cmd=$mode_cmd" --enable_bf16"
    fi
    if [[ ${CACHE_DIR} ]]; then
        mode_cmd=$mode_cmd" --load_dataset_from_file ${CACHE_DIR}"
    fi
    echo "tmp output, will remove later"${mode_cmd}

    python -u model_quant_convert.py \
            --model_name_or_path ${MODEL_NAME_OR_PATH} \
            --dataset_name ${DATASET} \
            --remove_unused_columns False \
            --do_eval \
            --no_cuda \
            --output_dir "model_and_tokenizer" \
            --overwrite_output_dir \
            --accuracy_only \
            --do_train \
            --per_device_eval_batch_size 8 \
            ${mode_cmd} 2>&1 | tee "$OUTPUT_DIR/$LOG_NAME-tune.log"
fi

echo "===== Model ${MODEL_NAME_OR_PATH} Inference with Mode ${MODE} ====== "
mode_cmd=""
if [[ ${PRECISION} = 'dynamic_int8' ]]; then
    mode_cmd=$mode_cmd" --dynamic_quantize True"
    inference_model="./model_and_tokenizer/fp32-model.onnx"
fi

if [[ ${MODE} == "accuracy" ]]; then
    echo "------------ACCURACY BENCHMARK---------"
    python run_executor.py \
      --input_model=${inference_model} \
      --mode=${MODE} \
      --batch_size=${BATCH_SIZE} \
      --warm_up=${WARM_UP} \
      --iteration=${ITERATION} \
      ${mode_cmd} 2>&1 | tee "$OUTPUT_DIR/$LOG_NAME-${MODE}-pipeline.log" 
    status=$?
    if [ ${status} != 0 ]; then
        echo "Benchmark process returned non-zero exit code."
        exit 1
    fi
elif [[ ${MODE} == "latency" ]]; then
    echo "------------LATENCY BENCHMARK---------"
    python run_executor.py \
      --input_model=${inference_model} \
      --mode="performance" \
      --batch_size=${BATCH_SIZE} \
      --warm_up=${WARM_UP} \
      --iteration=${ITERATION} \
      ${mode_cmd}  2>&1 | tee "$OUTPUT_DIR/$LOG_NAME-latency-pipeline.log" 
    status=$?
    if [ ${status} != 0 ]; then
        echo "Benchmark process returned non-zero exit code."
        exit 1
    fi
elif [[ ${MODE} == "throughput" ]]; then
    echo "------------MULTI-INSTANCE BENCHMARK---------"
    benchmark_cmd="python run_executor.py --input_model=${inference_model} --mode=performance --batch_size=${BATCH_SIZE} --warm_up=${WARM_UP} --iteration=${ITERATION} ${mode_cmd} "
    ncores_per_socket=${ncores_per_socket:=$( lscpu | grep 'Core(s) per socket' | cut -d: -f2 | xargs echo -n)}
    benchmark_pids=()
    ncores_per_instance=4
    export OMP_NUM_THREADS=${ncores_per_instance}
    logFile="$OUTPUT_DIR/$LOG_NAME-throughput"
    echo "Executing multi instance benchmark"
    for((j=0;$j<${ncores_per_socket};j=$(($j + ${ncores_per_instance}))));
    do
        end_core_num=$((j + ncores_per_instance -1))
        if [ ${end_core_num} -ge ${ncores_per_socket} ]; then
            end_core_num=$((ncores_per_socket-1))
        fi
        numactl -m 0 -C "$j-$end_core_num" \
        ${benchmark_cmd} 2>&1 | tee ${logFile}-${ncores_per_socket}-${ncores_per_instance}-${j}.log &
        benchmark_pids+=($!)
    done
    
    status="SUCCESS"
    for pid in "${benchmark_pids[@]}"; do
        wait $pid
        exit_code=$?
        echo "Detected exit code: ${exit_code}"
        if [ ${exit_code} == 0 ]; then
            echo "Process ${pid} succeeded"
        else
            echo "Process ${pid} failed"
            status="FAILURE"
        fi
    done
    echo "Benchmark process status: ${status}"
    if [ ${status} == "FAILURE" ]; then
        echo "Benchmark process returned non-zero exit code."
        exit 1
    fi
fi
