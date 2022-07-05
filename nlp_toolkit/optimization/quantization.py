from enum import Enum
from transformers.utils.versions import require_version

try:
    require_version("neural_compressor>=1.9.0")
except:
    require_version("neural_compressor_full>=1.9.0", "To fix: pip install neural_compressor")


class QuantizationMode(Enum):

    POSTTRAININGSTATIC = "post_training_static_quant"
    POSTTRAININGDYNAMIC = "post_training_dynamic_quant"
    QUANTIZATIONAWARETRAINING = "quant_aware_training"


SUPPORTED_QUANT_MODE = set([approach.name for approach in QuantizationMode])
