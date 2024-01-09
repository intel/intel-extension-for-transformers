import re
import functools
import inspect
from functools import partial
import torch
from torch.utils.checkpoint import checkpoint

def _get_relative_imports(module_file):
    with open(module_file, "r", encoding="utf-8") as f:
        content = f.read()
    relative_imports = re.findall(
        r"^\s*import\s+\.(\S+)\s*$", content, flags=re.MULTILINE
    )
    relative_imports += re.findall(
        r"^\s*from\s+\.(\S+)\s+import", content, flags=re.MULTILINE
    )
    relative_imports = set(relative_imports)
    # For Baichuan2
    if "quantizer" in relative_imports:
        relative_imports.remove("quantizer")

    return list(relative_imports)



def _gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    """
    Activates gradient checkpointing for the current model.

    Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
    activations".

    We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
    the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

    Args:
        gradient_checkpointing_kwargs (dict, *optional*):
            Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
    """
    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

    # For old GC format (transformers < 4.35.0) for models that live on the Hub
    # we will fall back to the overwritten `_set_gradient_checkpointing` methid
    _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

    if not _is_using_old_format:
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
    else:
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    if getattr(self, "_hf_peft_config_loaded", False):
        # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
        # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
        # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
        # the gradients to make sure the gradient flows.
        self.enable_input_require_grads()

def _gradient_checkpointing_disable(self):
    """
    Deactivates gradient checkpointing for the current model.

    Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
    activations".
    """
    if self.supports_gradient_checkpointing:
        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` methid
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=False)
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    if getattr(self, "_hf_peft_config_loaded", False):
        self.disable_input_require_grads()
