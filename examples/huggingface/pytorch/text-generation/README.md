# Text Generation
Text generation is a common task in natural language processing field. It leverages knowledge in computational linguistics and artificial intelligence to automatically generate natural language texts, which can satisfy certain communicative requirements.

We provide quantizatioin script [run_generation.py](./quantization/run_generation.py) for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B),  [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [decapoda-research/llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf), [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b), [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1), [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b), [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b) and [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b) .


We also provide Neural Engine Inference for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) and [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)


## Validated Models
Neural Compressor: 2.1

IPEX (Intel Extension for PyTorch): 2.0

Dataset: lambada-openai


| Model\Last token accuracy |  smoothquant config  | FP32  | BF16 | INT8 (mixed precision) |
|---------------------|:------:|:----------------------:|-----------------------|-----------------------------------|
| EleutherAI/gpt-j-6B | alpha 1.0/folding=False | 68.31% | 67.86% | 68.21% (w/o BF16) |
| decapoda-research/llama-7b-hf | alpha 0.8 | 73.61% | 73.26% | 73.57% (w/ FP32) |
| decapoda-research/llama-13b-hf | alpha 0.7 | 76.27% | 76.01% | 75.90% (w/ FP32) |
| decapoda-research/llama-30b-hf | alpha 0.7 | 77.57% | 77.53% | 78.40% (w/ FP32) |
| facebook/opt-125m   | alpha 0.5/folding=False | 37.9% | 37.63% | 37.57% (w/o BF16) |
| facebook/opt-350m   | alpha 0.8/folding=False | 45.16% | 45.06% | 45.53% (w/o BF16) |
| facebook/opt-2.7b   | alpha 0.5/folding=False | 63.65% | 63.23% | 64.04% (w/ BF16) |
| facebook/opt-6.7b   | alpha 0.5/folding=False | 67.69% | 67.36% | 68.04% (w/ BF16) |
| facebook/opt-13b   | alpha 0.5/folding=False | 68.72% | 67.84% | 68.14% (w/o BF16) |
| facebook/opt-30b   | alpha 0.5/folding=False | 71.49% | 70.87% | 71.28% (w/o BF16) |
| bigscience/bloom-560m   | alpha 0.5/folding=False | 35.4% | 25.56% | 35.36% (w/o BF16) |
| bigscience/bloom-1b7   | alpha 0.5/folding=False | 46.34% | 45.7% | 49.06% (w/ BF16) |
| bigscience/bloom-3b   | alpha 0.8/folding=False | 51.8% | 51.35% | 51.85% (w/o BF16) |
| bigscience/bloom-7b1   | alpha 0.5/folding=False | 57.64% | 57.23% | 59.77% (w/ BF16) |
| databricks/dolly-v1-6b   | alpha 0.8/folding=False | 68.66% | 67.96% | 68.95% (w/o BF16) |
| databricks/dolly-v2-3b   | alpha 0.5/folding=False | 62.97% | 60.86% | 62.47% (w/o BF16) |


