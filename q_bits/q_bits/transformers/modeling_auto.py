import torch
import transformers
from ..quantization_config import QBitsConfig
from ..utils import convert_to_quantized_model


class _BaseQBitsAutoModelClass:
    ORIG_MODEL = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        quantization_config = kwargs.pop("quantization_config", None)
        assert load_in_8bit or load_in_4bit or quantization_config is not None, \
            "QBitsAutoModelClass is used to load HF model and quantize it with weight-only!" \
            "So you should set load_in_8bit or load_in_4bit or quantization_config!"
        if load_in_4bit:
            if quantization_config is None:
                quantization_config = QBitsConfig(quant_bits=4, compute_dtype=torch_dtype, quant_dtype="s4fullrange")
            else:
                assert quantization_config.quant_bits == 4 and quantization_config.compute_dtype == torch_dtype, \
                f"Quantization_config.quant_bits should be 4 and compute_dtype should be {torch_dtype}."
        elif load_in_8bit:
            if quantization_config is None:
                quantization_config = QBitsConfig(quant_bits=8, compute_dtype=torch_dtype, quant_dtype="s8")
            else:
                assert quantization_config.quant_bits == 8 and quantization_config.compute_dtype == torch_dtype, \
                f"Quantization_config.quant_bits should be 8 and compute_dtype should be {torch_dtype}."
        elif quantization_config is not None:
            assert quantization_config.compute_dtype == torch_dtype, \
                f"Quantization_config.compute_dtype should be {torch_dtype}."

        model = cls.ORIG_MODEL.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return convert_to_quantized_model(model)


class AutoModelForCausalLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForCausalLM


class AutoModel(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModel


class AutoModelForSeq2SeqLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForSeq2SeqLM

