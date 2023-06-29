import logging
from typing import Dict, List, Optional, Type, Union

import torch
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.prompt.prompt_node import PromptModel, PromptModelInvocationLayer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

logger = logging.getLogger(__name__)


class TransformersDecoderInvocationLayer(PromptModelInvocationLayer):
    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        max_length: Optional[int] = 100,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        **kwargs,
    ):
        super().__init__(model_name_or_path, max_length)
        self.use_auth_token = use_auth_token

        self.devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        self.use_gpu = use_gpu
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )

        model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "model_kwargs",
                "trust_remote_code",
                "revision",
                "feature_extractor",
                "tokenizer",
                "config",
                "use_fast",
                "torch_dtype",
                "device_map",
            ]
            if key in kwargs
        }
        # flatten model_kwargs one level
        if "model_kwargs" in model_input_kwargs:
            mkwargs = model_input_kwargs.pop("model_kwargs")
            model_input_kwargs.update(mkwargs)

        torch_dtype = model_input_kwargs.get("torch_dtype")
        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if "torch." not in torch_dtype:
                    raise ValueError(
                        f"torch_dtype should be a torch.dtype or a string with 'torch.' prefix, got {torch_dtype}"
                    )
                torch_dtype_resolved = getattr(torch, torch_dtype.strip("torch."))
            elif isinstance(torch_dtype, torch.dtype):
                torch_dtype_resolved = torch_dtype
            else:
                raise ValueError(f"Invalid torch_dtype value {torch_dtype}")
            model_input_kwargs["torch_dtype"] = torch_dtype_resolved

        if len(model_input_kwargs) > 0:
            logger.info(
                "Using model input kwargs %s in %s", model_input_kwargs, self.__class__.__name__
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16
        )
        if self.use_gpu:
            self.model = self.model.to("cuda")

        # self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="sequential", torch_dtype=torch.float16)

        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model = self.model.eval()

    def invoke(self, *args, **kwargs):
        # generate config default values
        generate_kwargs = dict(
            use_cache=True,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )

        generate_kwargs.update(
            dict(
                min_new_tokens=kwargs.pop("min_new_tokens", 1),
                max_new_tokens=kwargs.pop("max_new_tokens", self.max_length),
                temperature=kwargs.pop("temperature", 0.9),
                top_p=kwargs.pop("top_p", 0.95),
                num_beams=kwargs.pop("beams", 1),
                early_stopping=kwargs.pop("early_stopping", False),
            )
        )
        decode_mode = (kwargs.pop("decode_mode", "Greedy"),)
        if "Greedy" in decode_mode:
            generate_kwargs.update(dict(beams=1, do_sample=False))
        elif "Beam" in decode_mode:
            generate_kwargs.update(dict(do_sample=True))
        else:
            pass
        generation_config = GenerationConfig(**generate_kwargs)

        output: List[Dict[str, str]] = []
        if kwargs and "prompt" in kwargs:
            prompt = kwargs.pop("prompt")
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            if inputs.get("token_type_ids") is not None:
                inputs.pop("token_type_ids")
            # inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to("cuda" if self.use_gpu else "cpu")
            with torch.no_grad():
                output = self.model.generate(**inputs, generation_config=generation_config)
                output = output[0][inputs["input_ids"].shape[-1] :]
        generated_texts = self.tokenizer.decode(output, skip_special_tokens=True)
        return [generated_texts]

    @classmethod
    def supports(cls, model_name_or_path: str) -> bool:
        try:
            config = AutoConfig.from_pretrained(model_name_or_path)
        except OSError:
            # This is needed so OpenAI models are skipped over
            return False
        supported_models = list(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values())
        ## un-official LLaMA models support HACK
        supported_models.append("LlamaForCausalLM")
        supported_models.append("LLaMAForCausalLM")
        return config.architectures[0] in supported_models


class FastRAGPromptModel(PromptModel):
    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        max_length: Optional[int] = 100,
        api_key: Optional[str] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.api_key = api_key
        self.use_auth_token = use_auth_token
        self.use_gpu = use_gpu
        self.devices = devices

        self.model_kwargs = model_kwargs if model_kwargs else {}

        self.invocation_layers: List[Type[PromptModelInvocationLayer]] = []

        self.register(TransformersDecoderInvocationLayer)

        self.model_invocation_layer = self.create_invocation_layer()

    def invoke(self, prompt: Union[str, List[str]], **kwargs) -> List[str]:
        output = self.model_invocation_layer.invoke(prompt=prompt, **kwargs, **self.model_kwargs)
        return output

