import logging
import torch.nn as nn
from accelerate import init_empty_weights
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from .nn import QuantizedLinearINT4, QuantizedLinearINT8


logger = logging.getLogger(__name__)


def _quantization_method(config):
    r"""
    This method returns the quantization method used for the model. If the model is not quantizable, it returns
    `None`.
    """
    if config.load_in_8bit:
        return "int8"
    elif config.load_in_4bit and config.bit4_quant_type == "int4":
        return "int4"
    else:
        return None


def replace_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    model, is_replaced = _replace_linear(
        model, modules_to_not_convert, current_key_name, quantization_config
    )

    if not is_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


def _replace_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, is_replaced=False
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    modules_to_not_convert = modules_to_not_convert.extend(quantization_config.llm_int8_skip_modules) if modules_to_not_convert else \
        quantization_config.llm_int8_skip_modules
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and modules_to_not_convert and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

                    if _quantization_method(quantization_config) == "int8":
                        model._modules[name] = QuantizedLinearINT8(
                            in_features,
                            out_features,
                            module.bias is not None,
                            compress_statistics=False,
                            blocksize=quantization_config.group_size,
                            scheme=quantization_config.scheme
                        )
                        is_replaced = True
                    else:
                        model._modules[name] = QuantizedLinearINT4(
                            in_features,
                            out_features,
                            module.bias is not None,
                            compress_statistics=False,
                            blocksize=quantization_config.group_size,
                            scheme=quantization_config.scheme
                        )
                        is_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
                    model._modules[name].set_weights(module.weight.data)
        if len(list(module.children())) > 0:
            _, is_replaced = _replace_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                is_replaced=is_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, is_replaced


def convert_to_quantized_model(model, config):
    conf = PostTrainingQuantConfig(
        approach='weight_only',
        op_type_dict={
            '.*':{
                "weight": {
                    'bits': config.quant_bits,
                    'group_size': config.group_size,  # -1 (per-channel)
                    'scheme': config.scheme, 
                    'algorithm': 'RTN', 
                },
            },
        },
    )
    model = quantization.fit(model, conf)
    replace_linear(model.model, None, None, config)
