import importlib
from transformers import AutoConfig, pipeline
from transformers.pipelines import *
from typing import Dict, Optional, Tuple
from .model import OptimizedModel


origin_forward = Pipeline.forward
origin_check = Pipeline.check_model_type

# pylint: disable=E0102
def infer_framework_load_model(
    model,
    config: AutoConfig,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    **model_kwargs
):
    """
    Support int8 model loading based on infer_framework_load_model

    Returns:
        `Tuple`: A tuple framework, model.
    """
    logger.warning("Function transformers.pipelines.base.infer_framework_load_model is replaced "
                    "by nlp_toolkit.optimization.pipeline.")

    backend = model_kwargs['backend'] if 'backend' in model_kwargs else None
    if isinstance(model, str):
        if backend == 'executor':
            from nlp_toolkit.backends.neural_engine.compile import compile
            model = compile(model)
            model.__call__= model.inference
            model.config = config
            framework = 'pt'

            # only support text-classification now
            def forward_executor(self, model_inputs, **forward_params):
                model_inputs = [v.int() for k, v in model_inputs.items()]
                model_outputs = model.inference(model_inputs)
                model_outputs = list(self.model.inference(model_inputs).values())[0]
                return {"logits": torch.from_numpy(model_outputs)}

            def check_model_type(self, supported_models):
                pass

            Pipeline.forward = forward_executor
            Pipeline.check_model_type = check_model_type
        else:
            model = OptimizedModel.from_pretrained(model, **model_kwargs)
            if hasattr(model, "eval"):
                model.eval()
            framework = "tf" if model.__class__.__name__.startswith("TF") else "pt"

            Pipeline.forward = origin_forward
            Pipeline.check_model_type = origin_check

    return framework, model


# Replace the function in pipeline to support int8 loading
trans_pipeline = importlib.import_module('transformers.pipelines')
trans_pipeline.infer_framework_load_model = infer_framework_load_model
