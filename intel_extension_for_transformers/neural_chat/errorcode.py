#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Error code and constant value for Neural Chat."""

STORAGE_THRESHOLD_GB = 30
GPU_MEMORY_THRESHOLD_MB = 6

class ErrorCodes:
    # General Service Error Code - System related
    ERROR_OUT_OF_MEMORY = 1001 # out of memory
    ERROR_DEVICE_BUSY = 1002 # device busy
    ERROR_DEVICE_NOT_FOUND = 1003 # device not exist
    ERROR_OUT_OF_STORAGE = 1004 # out of storage
    ERROR_DEVICE_NOT_SUPPORTED = 1005 # device not support
    ERROR_PLUGIN_NOT_SUPPORTED = 1006 # plugin not support

    # General Service Error Code - Model related
    ERROR_MODEL_NOT_FOUND = 2001
    ERROR_MODEL_CONFIG_NOT_FOUND = 2002
    ERROR_TOKENIZER_NOT_FOUND = 2003
    ERROR_CACHE_DIR_NO_WRITE_PERMISSION = 2004
    ERROR_INVALID_MODEL_VERSION = 2005
    ERROR_MODEL_NOT_SUPPORTED = 2006
    WARNING_INPUT_EXCEED_MAX_SEQ_LENGTH = 2101

    # General Service Error Code - Dataset related
    ERROR_DATASET_NOT_FOUND = 3001
    ERROR_DATASET_CONFIG_NOT_FOUND = 3002
    ERROR_VALIDATION_FILE_NOT_FOUND = 3003
    ERROR_TRAIN_FILE_NOT_FOUND = 3004
    ERROR_DATASET_CACHE_DIR_NO_WRITE_PERMISSION = 3005

    # Advanced Service Error Code - Finetune related
    ERROR_PTUN_FINETUNE_FAIL = 4001
    ERROR_LORA_FINETUNE_FAIL = 4002
    ERROR_LLAMA_ADAPTOR_FINETUNE_FAIL = 4003
    ERROR_PREFIX_FINETUNE_FAIL = 4004
    ERROR_PROMPT_FINETUNE_FAIL = 4005

    # Advanced Service Error Code - Inference related
    ERROR_WEIGHT_ONLY_QUANT_OPTIMIZATION_FAIL = 5001
    ERROR_AMP_OPTIMIZATION_FAIL = 5002
    ERROR_AUDIO_FORMAT_NOT_SUPPORTED = 5003
    ERROR_RETRIEVAL_DOC_FORMAT_NOT_SUPPORTED = 5004
    ERROR_SENSITIVE_CHECK_FILE_NOT_FOUND = 5005
    ERROR_MEMORY_CONTROL_FAIL = 5006
    ERROR_INTENT_DETECT_FAIL = 5007
    ERROR_MODEL_INFERENCE_FAIL = 5008
    ERROR_BITS_AND_BYTES_OPTIMIZATION_FAIL = 5009

    # General Service Error Code - Unknown Errors
    ERROR_GENERIC = 9999

    SUCCESS = 0  # The operation is executed successfully

    error_strings = {
        ERROR_OUT_OF_MEMORY: "System ran out of memory",
        ERROR_DEVICE_BUSY: "Device is currently busy",
        ERROR_DEVICE_NOT_FOUND: "Device does not exist",
        ERROR_OUT_OF_STORAGE: "System has run out of storage",
        ERROR_DEVICE_NOT_SUPPORTED: "Device is not supported",
        ERROR_PLUGIN_NOT_SUPPORTED: "Plugin is not supported",

        ERROR_MODEL_NOT_FOUND: "Requested model was not found",
        ERROR_MODEL_CONFIG_NOT_FOUND: "Model configuration not found",
        ERROR_TOKENIZER_NOT_FOUND: "Tokenizer not found",
        ERROR_CACHE_DIR_NO_WRITE_PERMISSION: "No write permission in cache directory",
        ERROR_INVALID_MODEL_VERSION: "Invalid model version",
        ERROR_MODEL_NOT_SUPPORTED: "Model is not supported",
        WARNING_INPUT_EXCEED_MAX_SEQ_LENGTH: "Input sequence exceeds maximum length",

        ERROR_DATASET_NOT_FOUND: "Dataset was not found",
        ERROR_DATASET_CONFIG_NOT_FOUND: "Dataset configuration not found",
        ERROR_VALIDATION_FILE_NOT_FOUND: "Validation file not found",
        ERROR_TRAIN_FILE_NOT_FOUND: "Training file not found",
        ERROR_DATASET_CACHE_DIR_NO_WRITE_PERMISSION: "No write permission in dataset cache directory",

        ERROR_PTUN_FINETUNE_FAIL: "PTUN finetuning failed",
        ERROR_LORA_FINETUNE_FAIL: "LORA finetuning failed",
        ERROR_LLAMA_ADAPTOR_FINETUNE_FAIL: "LLAMA Adaptor finetuning failed",
        ERROR_PREFIX_FINETUNE_FAIL: "Prefix finetuning failed",
        ERROR_PROMPT_FINETUNE_FAIL: "Prompt finetuning failed",

        ERROR_WEIGHT_ONLY_QUANT_OPTIMIZATION_FAIL: "Weight-only quantization optimization failed",
        ERROR_AMP_OPTIMIZATION_FAIL: "AMP optimization failed",
        ERROR_AUDIO_FORMAT_NOT_SUPPORTED: "Audio format is not supported",
        ERROR_RETRIEVAL_DOC_FORMAT_NOT_SUPPORTED: "Retrieval document format is not supported",
        ERROR_SENSITIVE_CHECK_FILE_NOT_FOUND: "Sensitive check file not found",
        ERROR_MEMORY_CONTROL_FAIL: "Memory control failed",
        ERROR_INTENT_DETECT_FAIL: "Intent detection failed",
        ERROR_MODEL_INFERENCE_FAIL: "Model inference failed",
        ERROR_BITS_AND_BYTES_OPTIMIZATION_FAIL: "Bits and bytes optimization failed",

        ERROR_GENERIC: "Generic error"
    }
