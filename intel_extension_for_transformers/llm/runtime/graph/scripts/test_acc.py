
import sys
from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate

model_name = sys.argv[1]
results = evaluate(
    model="hf-causal",
    model_args=f'pretrained="{model_name}"',
    tasks=["lambada_openai"],
    # limit=5,
    model_format="runtime"
)

print(results)