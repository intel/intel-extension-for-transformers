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
"""Neural Chat Chatbot API."""

from intel_extension_for_transformers.llm.quantization.optimization import Optimization
from .config import PipelineConfig
from .config import BaseFinetuningConfig
from .plugins import plugins

from .errorcode import ErrorCodes
from .utils.error_utils import set_latest_error, get_latest_error, clear_latest_error
from intel_extension_for_transformers.utils.logger import logging
import importlib

def check_tts_dependency():
    try:
        importlib.import_module('paddlespeech')
        importlib.import_module('paddle')
        importlib.import_module('soundfile')
        importlib.import_module('pydub')
        importlib.import_module('speechbrain')
        importlib.import_module('librosa')
        return True
    except ImportError:
        return False

def check_cache_dependency():
    try:
        importlib.import_module('gptcache')
        return True
    except ImportError:
        return False

def check_retrieval_dependency():
    try:
        importlib.import_module('fitz')
        importlib.import_module('easyocr')
        importlib.import_module('langchain')
        importlib.import_module('langchain_core')
        importlib.import_module('docx')
        importlib.import_module('bs4')
        importlib.import_module('unstructured')
        importlib.import_module('InstructorEmbedding')
        importlib.import_module('chromadb')
        importlib.import_module('openpyxl')
        return True
    except ImportError:
        return False

def check_faceanimation_dependency():
    try:
        importlib.import_module('face_alignment')
        importlib.import_module('imageio')
        importlib.import_module('resampy')
        importlib.import_module('kornia')
        importlib.import_module('tqdm')
        importlib.import_module('facexlib')
        importlib.import_module('gfpgan')
        importlib.import_module('av')
        importlib.import_module('safetensors')
        return True
    except ImportError:
        return False

def check_ner_dependency():
    try:
        importlib.import_module('spacy')
        importlib.import_module('pymysql')
        importlib.import_module('deepface')
        importlib.import_module('exifread')
        return True
    except ImportError:
        return False

def check_image2image_dependency():
    try:
        importlib.import_module('diffusers')
        return True
    except ImportError:
        return False


def build_chatbot(config: PipelineConfig=None):
    """Build the chatbot with a given configuration.

    Args:
        config (PipelineConfig): Configuration for building the chatbot.

    Returns:
        adapter: The chatbot model adapter.

    Example:
        from intel_extension_for_transformers.neural_chat import build_chatbot
        pipeline = build_chatbot()
        response = pipeline.predict(query="Tell me about Intel Xeon Scalable Processors.")
    """
    global plugins
    clear_latest_error()
    if not config:
        config = PipelineConfig()

    # create model adapter
    if "llama" in config.model_name_or_path.lower():
        from .models.llama_model import LlamaModel
        adapter = LlamaModel(config.model_name_or_path, config.task)
    elif "mpt" in config.model_name_or_path.lower():
        from .models.mpt_model import MptModel
        adapter = MptModel(config.model_name_or_path, config.task)
    elif "neural-chat" in config.model_name_or_path.lower():
        from .models.neuralchat_model import NeuralChatModel
        adapter = NeuralChatModel(config.model_name_or_path, config.task)
    elif "chatglm" in config.model_name_or_path.lower():
        from .models.chatglm_model import ChatGlmModel
        adapter = ChatGlmModel(config.model_name_or_path, config.task)
    elif "qwen" in config.model_name_or_path.lower():
        from .models.qwen_model import QwenModel
        adapter = QwenModel(config.model_name_or_path, config.task)
    elif "mistral" in config.model_name_or_path.lower():
        from .models.mistral_model import MistralModel
        adapter = MistralModel(config.model_name_or_path, config.task)
    elif "solar" in config.model_name_or_path.lower():
        from .models.solar_model import SolarModel
        adapter = SolarModel(config.model_name_or_path, config.task)
    elif "decilm" in config.model_name_or_path.lower():
        from .models.decilm_model import DeciLMModel
        adapter = DeciLMModel(config.model_name_or_path, config.task)
    elif "deepseek-coder" in config.model_name_or_path.lower():
        from .models.deepseek_coder_model import DeepseekCoderModel
        adapter = DeepseekCoderModel(config.model_name_or_path, config.task)
    elif "opt" in config.model_name_or_path.lower() or \
         "gpt" in config.model_name_or_path.lower() or \
         "flan-t5" in config.model_name_or_path.lower() or \
         "bloom" in config.model_name_or_path.lower() or \
         "starcoder" in config.model_name_or_path.lower() or \
         "codegen" in config.model_name_or_path.lower() or \
         "magicoder" in config.model_name_or_path.lower() or \
         "mixtral" in config.model_name_or_path.lower() or \
         "phi-2" in config.model_name_or_path.lower() or \
         "sqlcoder" in config.model_name_or_path.lower():
        from .models.base_model import BaseModel
        adapter = BaseModel(config.model_name_or_path, config.task)
    else:
        set_latest_error(ErrorCodes.ERROR_MODEL_NOT_SUPPORTED)
        logging.error("build_chatbot: unknown model")
        return
    from .models.base_model import register_model_adapter
    register_model_adapter(adapter)
    # register plugin instance in model adaptor
    if config.plugins:
        for plugin_name, plugin_value in config.plugins.items():
            enable_plugin = plugin_value.get('enable', False)
            if enable_plugin:
                if plugin_name == "tts" or plugin_name == "tts_chinese" or plugin_name == "asr":
                    if not check_tts_dependency():
                        raise ImportError(
                            f"Unable to initialize 'tts' plugin due to missing dependency packages.\n" \
                            f"Please run pip install -r requirements.txt to enable.\n" \
                            f"Please find the 'requirements.txt' file in the directory " \
                            "'intel_extension_for_transformers.neural_chat.pipeline.plugins.audio'."
                        )
                if plugin_name == "cache":
                    if not check_cache_dependency():
                        raise ImportError(
                            f"Unable to initialize 'cache' plugin due to missing dependency packages.\n" \
                            f"Please run pip install -r requirements.txt to enable.\n" \
                            f"Please find the 'requirements.txt' file in the directory " \
                            "'intel_extension_for_transformers.neural_chat.pipeline.plugins.caching'."
                        )
                if plugin_name == "retrieval":
                    if not check_retrieval_dependency():
                        raise ImportError(
                            f"Unable to initialize 'retrieval' plugin due to missing dependency packages.\n" \
                            f"Please run pip install -r requirements.txt to enable.\n" \
                            f"Please find the 'requirements.txt' file in the directory " \
                            "'intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval'."
                        )
                if plugin_name == "face_animation":
                    if not check_faceanimation_dependency():
                        raise ImportError(
                            f"Unable to initialize 'face_animation' plugin due to missing dependency packages.\n" \
                            f"Please run pip install -r requirements.txt to enable.\n" \
                            f"Please find the 'requirements.txt' file in the directory " \
                            "'intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation'."
                        )
                if plugin_name == "ner":
                    if not check_ner_dependency():
                        raise ImportError(
                            f"Unable to initialize 'ner' plugin due to missing dependency packages.\n" \
                            f"Please run pip install -r requirements.txt to enable.\n" \
                            f"Please find the 'requirements.txt' file in the directory. " \
                            f"'intel_extension_for_transformers.neural_chat.pipeline.plugins.ner'."
                        )
                if plugin_name == "image2image":
                    if not check_image2image_dependency():
                        raise ImportError(
                            f"Unable to initialize 'image2image' plugin due to missing dependency packages.\n" \
                            f"Please run pip install -r requirements.txt to enable.\n" \
                            f"Please find the 'requirements.txt' file in the directory " \
                            "'intel_extension_for_transformers.neural_chat.pipeline.plugins.image2image'."
                        )
                if plugin_name == "tts":
                    from .pipeline.plugins.audio.tts import TextToSpeech
                    plugins[plugin_name]['class'] = TextToSpeech
                elif plugin_name == "tts_chinese":
                    from .pipeline.plugins.audio.tts_chinese import ChineseTextToSpeech
                    plugins[plugin_name]['class'] = ChineseTextToSpeech
                elif plugin_name == "asr":
                    from .pipeline.plugins.audio.asr import AudioSpeechRecognition
                    plugins[plugin_name]['class'] = AudioSpeechRecognition
                elif plugin_name == "retrieval":
                    from .pipeline.plugins.retrieval.retrieval_agent import Agent_QA
                    plugins[plugin_name]['class'] = Agent_QA
                elif plugin_name == "cache":
                    from .pipeline.plugins.caching.cache import ChatCache
                    plugins[plugin_name]['class'] = ChatCache
                elif plugin_name == "safety_checker":
                    from .pipeline.plugins.security.safety_checker import SafetyChecker
                    plugins[plugin_name]['class'] = SafetyChecker
                elif plugin_name == "ner":
                    from .pipeline.plugins.ner.ner import NamedEntityRecognition
                    plugins[plugin_name]['class'] = NamedEntityRecognition
                elif plugin_name == "face_animation":
                    from .pipeline.plugins.video.face_animation.sadtalker import SadTalker
                    plugins[plugin_name]['class'] = SadTalker
                elif plugin_name == "image2image": # pragma: no cover
                    from .pipeline.plugins.image2image.image2image import Image2Image
                    plugins[plugin_name]['class'] = Image2Image
                else:
                    set_latest_error(ErrorCodes.ERROR_PLUGIN_NOT_SUPPORTED)
                    logging.error("build_chatbot: unknown plugin")
                    return
                print(f"create {plugin_name} plugin instance...")
                print(f"plugin parameters: ", plugin_value['args'])
                try:
                    plugins[plugin_name]["instance"] = plugins[plugin_name]['class'](**plugin_value['args'])
                except Exception as e:
                    if "[Rereieval ERROR] Document format not supported" in str(e):
                        set_latest_error(ErrorCodes.ERROR_RETRIEVAL_DOC_FORMAT_NOT_SUPPORTED)
                        logging.error("build_chatbot: retrieval plugin init failed")
                    elif "[SafetyChecker ERROR] Sensitive check file not found" in str(e):
                        set_latest_error(ErrorCodes.ERROR_SENSITIVE_CHECK_FILE_NOT_FOUND)
                        logging.error("build_chatbot: safety checker plugin init failed")
                    else:
                        set_latest_error(ErrorCodes.ERROR_GENERIC)
                        logging.error("build_chatbot: plugin init failed")
                    return
                adapter.register_plugin_instance(plugin_name, plugins[plugin_name]["instance"])

    parameters = {}
    parameters["model_name"] = config.model_name_or_path
    if config.tokenizer_name_or_path:
        parameters["tokenizer_name"] = config.tokenizer_name_or_path
    else:
        parameters["tokenizer_name"] = config.model_name_or_path
    parameters["device"] = config.device
    parameters["use_hpu_graphs"] = config.loading_config.use_hpu_graphs
    parameters["cpu_jit"] = config.loading_config.cpu_jit
    parameters["ipex_int8"] = config.loading_config.ipex_int8
    parameters["use_cache"] = config.loading_config.use_cache
    parameters["peft_path"] = config.loading_config.peft_path
    parameters["use_deepspeed"] = config.loading_config.use_deepspeed
    parameters["use_neural_speed"] = config.loading_config.use_neural_speed
    parameters["gguf_model_path"] = config.loading_config.gguf_model_path
    parameters["optimization_config"] = config.optimization_config
    parameters["hf_access_token"] = config.hf_access_token
    parameters["assistant_model"] = config.assistant_model
    if config.serving_config and config.serving_config.framework == "vllm":
        parameters["use_vllm"] = True
        parameters["vllm_engine_params"] = config.serving_config.framework_config
    else:
        parameters["use_vllm"] = False
        parameters["vllm_engine_params"] = None
    adapter.load_model(parameters)
    if get_latest_error():
        return
    else:
        return adapter

def finetune_model(config: BaseFinetuningConfig):
    """Finetune the model based on the provided configuration.

    Args:
        config (BaseFinetuningConfig): Configuration for finetuning the model.
    """
    clear_latest_error()
    assert config is not None, "BaseFinetuningConfig is needed for finetuning."
    from intel_extension_for_transformers.llm.finetuning.finetuning import Finetuning
    finetuning = Finetuning(config)
    try:
        finetuning.finetune()
    except FileNotFoundError as e:
        logging.error(f"Exception: {e}")
        if "Couldn't find a dataset script" in str(e):
            set_latest_error(ErrorCodes.ERROR_DATASET_NOT_FOUND)
    except ValueError as e:
        logging.error(f"Exception: {e}")
        if "--do_eval requires a validation dataset" in str(e):
            set_latest_error(ErrorCodes.ERROR_VALIDATION_FILE_NOT_FOUND)
        elif "--do_train requires a train dataset" in str(e):
            set_latest_error(ErrorCodes.ERROR_TRAIN_FILE_NOT_FOUND)
    except Exception as e:
        logging.error(f"Exception: {e}")
        if "Permission denied" in str(e):
            set_latest_error(ErrorCodes.ERROR_DATASET_CACHE_DIR_NO_WRITE_PERMISSION)
        elif config.finetune_args.peft == "lora":
            set_latest_error(ErrorCodes.ERROR_LORA_FINETUNE_FAIL)
        elif config.finetune_args.peft == "llama_adapter":
            set_latest_error(ErrorCodes.ERROR_LLAMA_ADAPTOR_FINETUNE_FAIL)
        elif config.finetune_args.peft == "ptun":
            set_latest_error(ErrorCodes.ERROR_PTUN_FINETUNE_FAIL)
        elif config.finetune_args.peft == "prefix":
            set_latest_error(ErrorCodes.ERROR_PREFIX_FINETUNE_FAIL)
        elif config.finetune_args.peft == "prompt":
            set_latest_error(ErrorCodes.ERROR_PROMPT_FINETUNE_FAIL)
        else:
            set_latest_error(ErrorCodes.ERROR_GENERIC)

def optimize_model(model, config, use_neural_speed=False):
    """Optimize the model based on the provided configuration.

    Args:
        model: large language model
        config (OptimizationConfig): The configuration required for optimizing the model.
        use_neural_speed (bool): A boolean indicating whether to use the LLM runtime graph optimization.
    """
    clear_latest_error()
    optimization = Optimization(optimization_config=config)
    try:
        model = optimization.optimize(model, use_neural_speed)
    except Exception as e:
        logging.error(f"Exception: {e}")
        from intel_extension_for_transformers.transformers import (
            MixedPrecisionConfig,
            WeightOnlyQuantConfig,
            BitsAndBytesConfig
        )
        if type(config) == MixedPrecisionConfig:
            set_latest_error(ErrorCodes.ERROR_AMP_OPTIMIZATION_FAIL)
        elif type(config) == WeightOnlyQuantConfig:
            set_latest_error(ErrorCodes.ERROR_WEIGHT_ONLY_QUANT_OPTIMIZATION_FAIL)
        elif type(config) == BitsAndBytesConfig:
            set_latest_error(ErrorCodes.ERROR_BITS_AND_BYTES_OPTIMIZATION_FAIL)
    return model
