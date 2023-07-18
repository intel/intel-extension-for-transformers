# Step-by-Step
We provide the inference benchmarking script `run_generation.py` for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B),  [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [decapoda-research/llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf), [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b), [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1), [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b), [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b), [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b), [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat), more models are working in progress.

>**Note**: The default search algorithm is beam search with num_beams = 4, if you'd like to use greedy search for comparison, add "--greedy" in args.


# Prerequisite​
## 1. Create Environment​
Recommend python 3.7 or higher version is recommended. The dependent packages are listed in requirements, please install them as follows,

```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt # Please source install intel-extension-for-pytorch before its 2.1 release.
```
Here is how to install intel-extension-for-pytorch from source.
```shell
#  gcc version >= 11
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git submodule sync && git submodule update --init --recursive
python setup.py install
```
We use the local GPTJ defination script `modeling_gptj.py` in `run_generation.py`. Here is a little change to success trace.
```diff
# Line 602 in modeling_gptj.py on transformers 4.28.1

-   position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
+   position_ids = torch.arange(past_length, torch.tensor(input_shape[-1]) + torch.tensor(past_length), dtype=torch.long, device=device)
```
The changes for `llama` series models in `modeling_llama.py`, `dolly_v2_3b` series models in `modeling_gpt_neox.py`， `bloom` series models in `modeling_bloom.py` and `opt` series models in `modeling_opt.py` are similar to the above.

`mosaicml/mpt-7b` has been updated frequently, and has not yet been integrated into `transformers`, so we fixed a commit number `68e1a8e0ebb9b30f3c45c1ef6195980f29063ae2` as a local folder to enable it.

# Run

## 1. Quantization
``` bash

python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --quantize \
    --sq \
    --alpha 1.0 \
    --int8_bf16_mixed \
    --ipex
```
## 2. Performance
```bash

python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --benchmark \
    --int8_bf16_mixed \
    --ipex
```
## 3. Accuracy
```bash

python run_generation.py \
   --model EleutherAI/gpt-j-6b \
   --accuracy \
   --int8_bf16_mixed \
   --ipex \
   --tasks "lambada_openai"
```
