#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import inspect
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from neural_compressor.model.torch_model import IPEXModel, PyTorchModel
from neural_compressor.utils.pytorch import load
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.auto_factory import _get_model_class
from transformers.utils.generic import ContextManagers
from optimum.exporters import TasksManager

from optimum.intel.neural_compressor import INCConfig
from optimum.intel.utils.import_utils import is_transformers_version
from optimum.modeling_base import OptimizedModel
from ..utils.utility import (
    DECODER_NAME,
    ENCODER_NAME,
    DECODER_WITH_PAST_NAME
)

if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin

logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Base INCModelForSeq2SeqLM class.
    """,
)
class INCBaseModelForSeq2SeqLM(OptimizedModel):
    _AUTOMODELS_TO_TASKS = {cls_name: task for task, cls_name in TasksManager._TASKS_TO_AUTOMODELS.items()}
    base_model_prefix = "inc_model"
    auto_model_class = AutoModel
    export_feature = "text2text-generation"

    def __init__(
        self,
        encoder_model,
        decoder_model,
        decoder_with_past_model=None,
        config: PretrainedConfig=None,
        device: str = "CPU",
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self.config = config
        self.model_save_dir = model_save_dir
        self._device = device.upper()
        self.preprocessors = kwargs.get("preprocessors", [])

        self._encoder_model = encoder_model
        self._decoder_model = decoder_model
        self._decoder_with_past_model = decoder_with_past_model

        if is_transformers_version("<=", "4.25.1"):
            self.generation_config = None
        else:
            from transformers import GenerationConfig

            self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    @staticmethod
    def load_model(file_name: Union[str, Path]):
        model = torch.jit.load(file_name)
        torch.jit.freeze(model.eval())
        return model

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        encoder_file_name: Optional[str] = ENCODER_NAME,
        decoder_file_name: Optional[str] = DECODER_NAME,
        decoder_with_past_file_name: Optional[str] = DECODER_WITH_PAST_NAME,
        **kwargs,
    ):
        """
        Saves the model so that it can be re-loaded using the
        [`from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
            encoder_file_name(`str`, *optional*):
                The encoder model file name. Overwrites the default file name and allows one to save the encoder model
                with a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name. Overwrites the default file name and allows one to save the decoder model
                with a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name, allowing to save
                the decoder model with a different name.
        """
        if isinstance(self.encoder_model, IPEXModel):
            self.encoder_model.model.save(os.path.join(save_directory, encoder_file_name))
            self.decoder_model.model.save(os.path.join(save_directory, decoder_file_name))
            if self.decoder_with_past_model is not None:
                self.decoder_with_past_model.model.save(os.path.join(save_directory, decoder_with_past_file_name))
        elif isinstance(self.encoder_model, PyTorchModel):
            encoder_state_dict = self.encoder_model._model.state_dict()
            decoder_state_dict = self.decoder_model._model.state_dict()
            if self.decoder_with_past_model is not None:
                decoder_with_past_state_dict = self.decoder_with_past_model._model.state_dict()

            if hasattr(self.encoder_model, "q_config"):
                encoder_state_dict["best_configure"] = self.encoder_model.q_config
            torch.save(encoder_state_dict, os.path.join(save_directory, encoder_file_name))
            if hasattr(self.decoder_model, "q_config"):
                decoder_state_dict["best_configure"] = self.decoder_model.q_config
            torch.save(decoder_state_dict, os.path.join(save_directory, decoder_file_name))
            if self.decoder_with_past_model is not None:
                if hasattr(self.decoder_with_past_model, "q_config"):
                    decoder_with_past_state_dict["best_configure"] = self.decoder_with_past_model.q_config
                torch.save(decoder_with_past_state_dict, os.path.join(save_directory, decoder_with_past_file_name))
        elif self.config.torchscript:
            torch.jit.save(self.encoder_model, os.path.join(save_directory, encoder_file_name))
            torch.jit.save(self.decoder_model, os.path.join(save_directory, decoder_file_name))
            if self.decoder_with_past_model is not None:
                torch.jit.save(self.decoder_with_past_model, os.path.join(save_directory, decoder_with_past_file_name))
        else:
            encoder_state_dict = self.encoder_model.state_dict()
            decoder_state_dict = self.decoder_model.state_dict()

            torch.save(encoder_state_dict, os.path.join(save_directory, encoder_file_name))
            torch.save(decoder_state_dict, os.path.join(save_directory, decoder_file_name))
        logger.info(f"Model weights saved to {save_directory}")

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        encoder_file_name: Optional[str] = ENCODER_NAME,
        decoder_file_name: Optional[str] = DECODER_NAME,
        decoder_with_past_file_name: Optional[str] = DECODER_WITH_PAST_NAME,
        local_files_only: bool = False,
        use_cache: bool = True,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            config (PretrainedConfig):
                model configure
            use_auth_token (`str` or `bool`):
                The token to use as HTTP bearer authorization for remote files. Needed to load models from a private
                repository.
            revision (`str`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            encoder_file_name(`str`, *optional*):
                The encoder model file name. Overwrites the default file name and allows one to
                load the encoder model with a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name. Overwrites the default file name and allows one to
                load the decoder model with a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name,
                allowing to load the decoder model with a different name.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models).
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
                (`torch.bfloat16`, or `"auto"`).
        """
        model_dim = None
        model_parallel = None

        decoder_with_past = None
        try:
            inc_config = INCConfig.from_pretrained(model_id)
        except Exception:
            logger.info("Couldn't find inc config.")
            inc_config = None
        if config.torchscript:
            # Load model from a local directory
            if os.path.isdir(model_id):
                encoder = cls.load_model(os.path.join(model_id, encoder_file_name))
                decoder = cls.load_model(os.path.join(model_id, decoder_file_name))
                decoder_with_past = (
                    cls.load_model(os.path.join(model_id, decoder_with_past_file_name)) if use_cache else None
                )

            # Load model from hub
            else:
                model_file_names = {"encoder": encoder_file_name, "decoder": decoder_file_name}
                if use_cache:
                    model_file_names["decoder_with_past"] = decoder_with_past_file_name

                try:
                    for name, file_name in model_file_names.items():
                        # pylint: disable=E1123
                        model_cache_path = hf_hub_download(
                            repo_id=model_id,
                            filename=file_name,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            local_files_only=local_files_only,
                        )
                        model_file_names[name] = model_cache_path
                except Exception:
                    logger.warning(
                        "The file names `encoder_model.bin` and `decoder_model.bin` was not found! Please check it!"
                    )
                    raise Exception

                encoder = cls.load_model(model_file_names["encoder"])
                decoder = cls.load_model(model_file_names["decoder"])
                decoder_with_past = cls.load_model(model_file_names["decoder_with_past"]) if use_cache else None
        else:
            model_kwargs = {
                "revision": revision,
                "use_auth_token": use_auth_token,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
                "force_download": force_download,
                "torch_dtype": torch_dtype,
            }
            task = cls.export_feature
            if inc_config is None:
                model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
                encoder = model.encoder
                decoder = model
            else:
                model_class = _get_model_class(config, cls.auto_model_class._model_mapping)
                init_contexts = [no_init_weights(_enable=True)]
                with ContextManagers(init_contexts):
                    model = model_class(config)

                # Load the model from local directory
                if os.path.isdir(model_id):
                    encoder_state_dict_path = os.path.join(model_id, encoder_file_name)
                    decoder_state_dict_path = os.path.join(model_id, decoder_file_name)
                # Download the model from the hub
                else:
                    encoder_state_dict_path = hf_hub_download(  # pylint: disable=E1123
                        repo_id=model_id,
                        filename=encoder_file_name,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                    decoder_state_dict_path = hf_hub_download(  # pylint: disable=E1123
                        repo_id=model_id,
                        filename=decoder_file_name,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                # Load the state dictionary of the model to verify whether the model is quantized or not
                encoder_state_dict = torch.load(encoder_state_dict_path, map_location="cpu")
                decoder_state_dict = torch.load(decoder_state_dict_path, map_location="cpu")
                if (
                    "best_configure" in encoder_state_dict
                    and encoder_state_dict["best_configure"] is not None
                    and "best_configure" in decoder_state_dict
                    and decoder_state_dict["best_configure"] is not None
                ):
                    try:
                        encoder = load(encoder_state_dict_path, model.encoder)
                        decoder = load(decoder_state_dict_path, model)
                    except Exception as e:
                        logger.error(e.args)
                        raise
                else:
                    encoder = model.encoder.load_state_dict(encoder_state_dict)
                    decoder = model.load_state_dict(decoder_state_dict)
            model_dim = model.model_dim
            model_parallel = model.model_parallel

        return cls(
            encoder_model=encoder,
            decoder_model=decoder,
            decoder_with_past_model=decoder_with_past,
            config=config,
            use_cache=use_cache,
            model_dim=model_dim,
            model_parallel=model_parallel,
            **kwargs,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        task: Optional[str] = None,
        use_cache: bool = True,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        **kwargs,
    ):
        """
        Export a vanilla Transformers model into an JIT model.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            config (PretrainedConfig):
                model configure
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, will only try to load the tokenizer configuration from local files.
            task (str):
                Task name in _TASK_LEGACY which defined in
                https://github.com/huggingface/optimum-intel/optimum/intel/utils/constant.py#LL27C1-L27C13
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models).
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
                (`torch.bfloat16`, or `"auto"`).
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        encoder_file_name = os.path.join("encoder", ENCODER_NAME)
        decoder_file_name = os.path.join("decoder", DECODER_NAME)
        decoder_with_past_file_name = os.path.join("decoder_with_past", DECODER_WITH_PAST_NAME)
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
            "torch_dtype": torch_dtype,
        }

        model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)

        model.config.return_dict = False
        model.encoder.config.return_dict = False
        model.decoder.config.return_dict = False
        encoder_signature = inspect.signature(model.encoder.forward) \
            if hasattr(model.encoder, "forward") else inspect.signature(model.encoder.call)
        decoder_signature = inspect.signature(model.forward) \
            if hasattr(model, "forward") else inspect.signature(model.call)
        onnx_config_class = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
        onnx_config = onnx_config_class(model.config, use_past=use_cache)
        encoder_onnx_config = onnx_config.with_behavior("encoder")
        decoder_onnx_config = onnx_config.with_behavior("decoder", use_past=False)
        encoder_dummy_inputs = encoder_onnx_config.generate_dummy_inputs(framework="pt")
        decoder_dummy_inputs = decoder_onnx_config.generate_dummy_inputs(framework="pt")
        encoder_model_inputs = {
            key: encoder_dummy_inputs[key] for key in encoder_signature.parameters \
                if encoder_dummy_inputs.get(key, None) is not None
        }
        decoder_model_inputs = {
            key: decoder_dummy_inputs[key] for key in decoder_signature.parameters \
                if decoder_dummy_inputs.get(key, None) is not None
        }
        decoder_model_inputs["encoder_outputs"] = (decoder_model_inputs["encoder_outputs"][0:1][0].to(torch_dtype),)
        encoder_traced_model = torch.jit.trace(model.encoder, example_kwarg_inputs=encoder_model_inputs)
        decoder_traced_model = torch.jit.trace(model, example_kwarg_inputs=decoder_model_inputs)
        encoder_traced_model = torch.jit.freeze(encoder_traced_model.eval())
        decoder_traced_model = torch.jit.freeze(decoder_traced_model.eval())
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        os.makedirs(os.path.join(save_dir_path, "encoder"))
        os.makedirs(os.path.join(save_dir_path, "decoder"))
        torch.jit.save(encoder_traced_model, os.path.join(save_dir_path, encoder_file_name))
        torch.jit.save(decoder_traced_model, os.path.join(save_dir_path, decoder_file_name))
        if use_cache:
            decoder_with_past_onnx_config = onnx_config.with_behavior("decoder", use_past=True)
            decoder_with_past_dummy_inputs = decoder_with_past_onnx_config.generate_dummy_inputs(framework="pt")
            decoder_with_past_model_inputs = {
                key: decoder_with_past_dummy_inputs[key] for key in decoder_signature.parameters \
                    if decoder_with_past_dummy_inputs.get(key, None) is not None
            }
            decoder_with_past_model_inputs["encoder_outputs"] = \
                (decoder_with_past_model_inputs["encoder_outputs"][0:1][0].to(torch_dtype),)
            pkv = []
            for i in range(len(decoder_with_past_model_inputs['past_key_values'])):
                pkv.append([])
                for j in range(len(decoder_with_past_model_inputs['past_key_values'][0])):
                    pkv[i].append(decoder_with_past_model_inputs['past_key_values'][i][j].to(torch_dtype))
                pkv[i] = tuple(pkv[i])
            decoder_with_past_model_inputs['past_key_values'] = tuple(pkv)
            decoder_with_past_traced_model = \
                torch.jit.trace(model, example_kwarg_inputs=decoder_with_past_model_inputs)
            decoder_with_past_traced_model = torch.jit.freeze(decoder_with_past_traced_model.eval())
            os.makedirs(os.path.join(save_dir_path, "decoder_with_past"))
            torch.jit.save(decoder_with_past_traced_model, os.path.join(save_dir_path, decoder_with_past_file_name))

        config.torchscript = True

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_cache=use_cache,
            from_onnx=True,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            encoder_file_name=encoder_file_name,
            decoder_file_name=decoder_file_name,
            decoder_with_past_file_name=decoder_with_past_file_name,
            local_files_only=local_files_only,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _auto_model_to_task(cls, auto_model_class):
        """
        Get the task corresponding to a class (for example AutoModelForXXX in transformers).
        """
        return cls._AUTOMODELS_TO_TASKS[auto_model_class.__name__]

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        """
        if isinstance(self, GenerationMixin):
            return True
        return False

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.encoder.to(self._device)
        self.decoder.to(self._device)
        return self

    def eval(self):
        self.model.eval()
