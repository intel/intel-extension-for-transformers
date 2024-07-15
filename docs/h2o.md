# H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
1. [Introduction](#introduction)
2. [Usage](#usage)

## Introduction
**Heavy-Hitter Oracal (H2O)** is a novel approach for implementing the KV cache which significantly reduces memory footprint. 

This methods base on the fact that the accumulated attention scores of all tokens in attention blocks adhere to a power-law distribution. It suggests that there exists a small set of influential tokens that are critical during generation, named heavy-hitters (H2). H2 provides an opportunity to step away from the combinatorial search problem and identify an eviction policy that maintains accuracy.

H2O can dynamically retains the balance of recent and H2 tokens. Significantly increase model throughput while ensuring accuracy.


For more info, please refer to the paper [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/pdf/2306.14048).


![](./imgs/h2o.png)


## Usage
Using simulation mode
```python
from intel_extension_for_transformers.transformers.kv_cache_compression import H2OConfig, LlamaForCausalLM
h2o_config = H2OConfig(
    heavy_ratio=heavy_ratio,
    recent_ratio=recent_ratio,
    h2o_min_seqlen=h2o_min_seqlen,
    real_drop=False,
)
user_model = LlamaForCausalLM.from_pretrained(
    args.model,
    prune_config=h2o_config,
    trust_remote_code=args.trust_remote_code)
```
To run the real_drop mode
```python
from intel_extension_for_transformers.transformers.kv_cache_compression import H2OConfig, LlamaForCausalLM
h2o_config = H2OConfig(
    heavy_ratio=heavy_ratio,
    recent_ratio=recent_ratio,
    h2o_min_seqlen=h2o_min_seqlen,
    real_drop=True,
)
user_model = LlamaForCausalLM.from_pretrained(
    args.model,
    prune_config=h2o_config,
    trust_remote_code=args.trust_remote_code)
```

Please refer to [h2o example](../examples/huggingface/pytorch/text-generation/h2o/run_generation.py) for the details.
