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

MEMORY_THRESHOLD_GB = 8
STORAGE_THRESHOLD_GB = 30
GPU_MEMORY_THRESHOLD_MB = 6

class ResponseCodes:
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

    # General Service Error Code - Unknown Errors
    ERROR_GENERIC = 9999

    SUCCESS = 0  # The operation is executed successfully

    @property
    def string(self):
        return self.error_strings.get(self, "Unknown Error")

    error_strings = {
        ERROR_OUT_OF_MEMORY: "Out of Memory",
        ERROR_DEVICE_BUSY: "Device Busy",
        ERROR_DEVICE_NOT_FOUND: "Device Not Exist",
        ERROR_OUT_OF_STORAGE: "Out of Storage",
        ERROR_DEVICE_NOT_SUPPORTED: "Device Not Supported",
        ERROR_MODEL_NOT_FOUND: "Model Not Found",
        ERROR_MODEL_CONFIG_NOT_FOUND: "Model Config Not Found",
        ERROR_TOKENIZER_NOT_FOUND: "Tokenizer Not Found",
        ERROR_CACHE_DIR_NO_WRITE_PERMISSION: "Cache Directory No Write Permission",
        ERROR_INVALID_MODEL_VERSION: "Invalid Model Version",
        ERROR_MODEL_NOT_SUPPORTED: "Model Not Supported",
        WARNING_INPUT_EXCEED_MAX_SEQ_LENGTH: "Input Exceeds Max Sequence Length",
        ERROR_DATASET_NOT_FOUND: "Dataset Not Found",
        ERROR_DATASET_CONFIG_NOT_FOUND: "Dataset Config Not Found",
        ERROR_VALIDATION_FILE_NOT_FOUND: "Validation File Not Found",
        ERROR_TRAIN_FILE_NOT_FOUND: "Train File Not Found",
        ERROR_DATASET_CACHE_DIR_NO_WRITE_PERMISSION: "Dataset Cache Directory No Write Permission",
        ERROR_PTUN_FINETUNE_FAIL: "PTUN Finetune Fail",
        ERROR_LORA_FINETUNE_FAIL: "LORA Finetune Fail",
        ERROR_LLAMA_ADAPTOR_FINETUNE_FAIL: "LLAMA Adaptor Finetune Fail",
        ERROR_PREFIX_FINETUNE_FAIL: "Prefix Finetune Fail",
        ERROR_PROMPT_FINETUNE_FAIL: "Prompt Finetune Fail",
        ERROR_WEIGHT_ONLY_QUANT_OPTIMIZATION_FAIL: "Weight Only Quant Optimization Fail",
        ERROR_AMP_OPTIMIZATION_FAIL: "AMP Optimization Fail",
        ERROR_AUDIO_FORMAT_NOT_SUPPORTED: "Audio Format Not Supported",
        ERROR_RETRIEVAL_DOC_FORMAT_NOT_SUPPORTED: "Retrieval Document Format Not Supported",
        ERROR_SENSITIVE_CHECK_FILE_NOT_FOUND: "Sensitive Check File Not Found",
        ERROR_MEMORY_CONTROL_FAIL: "Memory Control Fail",
        ERROR_INTENT_DETECT_FAIL: "Intent Detection Fail",
        ERROR_MODEL_INFERENCE_FAIL: "Model Inference Fail",
        ERROR_GENERIC: "Generic Error"
    }