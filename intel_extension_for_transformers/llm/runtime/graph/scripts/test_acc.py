
from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
# test evaluate encoder_model + decoder_model_merged
results = evaluate(
    model="hf-causal",
    model_args='pretrained="/mnt/disk1/data2/zhenweil/models/gptq/Llama-2-7B-Chat-GPTQ",dtype=float32',
    tasks=["lambada_openai"],
    limit=5,
    model_format="runtime"
)

print(results)