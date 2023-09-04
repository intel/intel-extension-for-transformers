import torch
import transformers
from ...llm.quantization.config import WeightOnlyConfig
from ...llm.quantization.utils import convert_to_quantized_model

from .gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM
# to use modeling modification base transformers 4.30.2:
transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTJForCausalLM = GPTBigCodeForCausalLM

# Will remove after torch 2.1 release
# to use modeling modification base transformers 4.28.1:
from .gptj.modeling_gptj import GPTJForCausalLM
from .llama.modeling_llama import LlamaForCausalLM
from .bloom.modeling_bloom import BloomForCausalLM
from .gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from .opt.modeling_opt import OPTForCausalLM
transformers.models.gptj.modeling_gptj.GPTJForCausalLM = GPTJForCausalLM
transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM
transformers.models.bloom.modeling_bloom.BloomForCausalLM = BloomForCausalLM
transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM = GPTNeoXForCausalLM
transformers.models.opt.modeling_opt.OPTForCausalLM = OPTForCausalLM

class _BaseQBitsAutoModelClass:
    ORIG_MODEL = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)
        if load_in_8bit or load_in_4bit or quantization_config is not None:
            torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        if load_in_4bit:
            if quantization_config is None:
                quantization_config = WeightOnlyConfig(compute_dtype=torch_dtype, weight_dtype="nf4")
            else:
                assert "4" in quantization_config.weight_dtype and quantization_config.compute_dtype == torch_dtype, \
                f"Quantization_config.weight_dtype should be 'nf4', 'int4_fullrange', 'int4_clip',"
                f"'fp4_e2m1', 'fp4_e2m1_bnb' or 'int8' and compute_dtype should be {torch_dtype}."
        elif load_in_8bit:
            if quantization_config is None:
                quantization_config = WeightOnlyConfig(compute_dtype=torch_dtype, weight_dtype="int8")
            else:
                assert quantization_config.weight_dtype == "int8" and quantization_config.compute_dtype == torch_dtype, \
                f"Quantization_config.weight_dtype should be 'int8' and compute_dtype should be {torch_dtype}."
        elif quantization_config is not None:
            assert quantization_config.compute_dtype == torch_dtype, \
                f"Quantization_config.compute_dtype should be {torch_dtype}."

        model = cls.ORIG_MODEL.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if quantization_config is not None:
            return convert_to_quantized_model(model, quantization_config)
        else:
            return model


class AutoModelForCausalLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForCausalLM


class AutoModel(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModel


class AutoModelForSeq2SeqLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForSeq2SeqLM

