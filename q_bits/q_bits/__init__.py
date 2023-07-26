import os
import torch
from .quantization_config import QBitsConfig
from .utils import convert_to_quantized_model

torch.ops.load_library(os.path.join(os.path.abspath(os.path.dirname(__file__)), "libweight_only_jblasop.so"))
