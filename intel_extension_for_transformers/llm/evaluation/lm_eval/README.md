[`lm_eval`](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) provides a unified framework to test generative language models on a large number of different evaluation tasks. It recommends wrapper model based on the `LM` class usage, so [`HFCausalLM`](./lm_model.py), [`HFSeq2SeqLM`](./lm_model.py) is created to enable language models. Commit id `2c18e367c6ded428863cd1fd4cf9558ca49d68dc` is used for lm_eval to make results stable.
## Usages
### PyTorch example
`model` and `tokenizer` are necessary parameters for PyTorch, Please provide `config` for some special cases, for example, the model doesn't attribute config. 
```bash
from intel_extension_for_transformers.llm.evaluation.lm_eval import HFCausalLM, HFSeq2SeqLM, evaluate
clm_tokenizer =  AutoTokenizer.from_pretrained("facebook/opt-125m")
clm_model = HFCausalLM(model=clm_model, tokenizer=clm_tokenizer)
seq2seq_tokenizer = AutoTokenizer.from_pretrained("t5-small")
seq2seq_model = HFCausalLM(model=seq2seq_model, tokenizer=seq2seq_tokenizer)
results = evaluate(
    model=clm_model,
    tasks=["piqa"],
    batch_size=1,
    limit=20,
    no_cache=True
)
results = evaluate(
    model=seq2seq_model,
    tasks=["piqa"],
    batch_size=1
    limit=20,
    no_cache=True
)
```

### ONNX example
`model_name_or_path` is necessary parameter for `ONNX`, Please sure tokenizer config files in the folder if you don't provide tokenizer.
```bash
from intel_extension_for_transformers.llm.evaluation.lm_eval import HFCausalLM, HFSeq2SeqLM, evaluate
model = HFCausalLM(model_name_or_path="./gptj", model_format="onnx")   
results = evaluate(
    model= model,
    tasks=["piqa"],
    limit=20,
    no_cache=True
)

model = HFSeq2SeqLM(model_name_or_path="./t5-past", model_format="onnx")
results = evaluate(
    model=model,
    tasks=["piqa"],
    limit=20,
    no_cache=True
)
```