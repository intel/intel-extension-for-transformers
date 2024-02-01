Step-by-Step
============

Please follow Intel® Neural Compressor [document](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager) to prune Huggingface Large Language Models(LLMs)

## Prerequisite
### Create Environment
```bash
# Create Environment (conda)
conda create -n llm python=3.9 -y
conda install mkl mkl-include -y
conda install gperftools jemalloc==5.2.1 -c conda-forge -y

# Installation
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
pip install -r requirements.txt
pip install -v .
cd examples/huggingface/pytorch/language-modeling/pruning
pip install -r requirements.txt
pip install transformers==4.34.1
```
>**Note**: Please use transformers no higher than 4.34.1

## Retrain-free Results

The last word acc of the channel-wise sparse model using [the retrain-free scripts](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/language-modeling/pruning/scripts/run_gptj_pruning.sh) is shown in the following table. All the sparsity is 10% over MLP block.

| Model | Task | Calibration dataset | Evaluation dataset | Precision | Dense last word accuracy | Sparse last word accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:|
| EleutherAI/gpt-j-6b | CLM | pile_10k | lambada_openai | FP32 | 0.6831 | 0.6819 | -0.17% |
| EleutherAI/gpt-j-6b | CLM | pile_10k | lambada_openai | BF16 | 0.6792 | 0.6767 | -0.36% |
| facebook/opt-1.3b | CLM | pile_10k | lambada_openai | FP32 | 0.5789 |0.5686  | -1.73% |
| facebook/opt-1.3b | CLM | pile_10k | lambada_openai | BF16 | 0.5629 | 0.5501 | -1.78% |
| facebook/opt-2.7b | CLM | pile_10k | lambada_openai | FP32 | 0.6365 | 0.6367 | +0.03% |
| facebook/opt-2.7b | CLM | pile_10k | lambada_openai | BF16 | 0.6336 | 0.6344 | +0.12% |
| decapoda-research/llama-7b-hf | CLM | pile_10k | lambada_openai | FP32 | 0.7361 | 0.7298 | -0.86% |
| decapoda-research/llama-7b-hf | CLM | pile_10k | lambada_openai | BF16 | 0.7326 | 0.7271 | -0.75% |
| bigscience/bloom-1b7 | CLM | pile_10k | lambada_openai | FP32 | 0.4634 | 0.4636 | 0.04% |
| bigscience/bloom-1b7 | CLM | pile_10k | lambada_openai | BF16 | 0.4570 | 0.4572 | 0.04% |
| bigscience/bloom-7b1 | CLM | pile_10k | lambada_openai | FP32 | 0.5764 | 0.5791 | 0.47% |
| bigscience/bloom-7b1 | CLM | pile_10k | lambada_openai | BF16 | 0.5723 | 0.5756 | 0.58% |


## SparseGPT Results

The last word acc of the 1x1 pattern sparse model using [the sparseGPT script](examples/huggingface/pytorch/language-modeling/pruning/scripts/run_llm_sparseGPT.sh) is shown in the following table.
| Model | Task | Calibration dataset | Evaluation dataset | Sparsity | Precision | Dense last word accuracy | Sparse last word accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:|
| meta-llama/Llama-2-7b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 30% | FP32 | 0.7392 | 0.7320 | -0.97% |
| meta-llama/Llama-2-7b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 30% | BF16 | 0.7365 | 0.7304 | -1.19% |
| EleutherAI/gpt-j-6b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6831 | 0.6922 | +1.33% |
| EleutherAI/gpt-j-6b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6771 | 0.6874 | +0.63% |
| decapoda-research/llama-7b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7361 | 0.7332 | -0.39% |
| decapoda-research/llama-7b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.7326 | 0.7297 | -0.87% |
| facebook/opt-6.7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6769 | 0.6616 | -2.26% |
| facebook/opt-6.7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6730 | 0.6577 | -2.84% |
| tiiuae/falcon-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7467 | 0.7528 | +0.82% |
| tiiuae/falcon-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.7464 | 0.7502 | +0.47% |
| bigscience/bloom-7b1 | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.5764 | 0.5606 | -2.74% |
| bigscience/bloom-7b1 | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.5725 | 0.5587 | -3.07% |
| mosaicml/mpt-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7056 | 0.7035 | -0.30% |
| mosaicml/mpt-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6831 | 0.6856 | -2.83% |
| mosaicml/mpt-7b-chat | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6550 | 0.6561 | +0.17% |
| mosaicml/mpt-7b-chat | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6456 | 0.6451 | -1.51% |
| meta-llama/Llama-2-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7679 | 0.7629 | -0.65% |
| meta-llama/Llama-2-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.7667 | 0.7601 | -1.02% |
| decapoda-research/llama-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 50% | FP32 | 0.7627 | 0.7559 | -0.89% |
| decapoda-research/llama-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 50% | BF16 | 0.7599 | 0.7559 | -0.89% |


