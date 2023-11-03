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
```

## Retrain-free Results

The last token accuracy for channel pruning using [the retrain-free scripts](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/language-modeling/pruning/scripts/run_gptj_pruning.sh) is presented in the following table.
| Model | Calibration dataset | Evaluation dataset | Sparsity pattern | Over MLP block sparsity |Element-wise/matmul, Gemm, conv ratio | Dense last token accuracy | Sparse last token accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:| :----: |
| EleutherAI/gpt-j-6b | lambada | lambada | channelx1  | 0.1999 | 0.1242 | 0.7917 | 0.8038 | +1.50% |
| EleutherAI/gpt-j-6b | the_pile | lambada |  channelx1  | 0.0999 | 0.0643 | 0.7917 | 0.7931 | +0.17% |
| EleutherAI/gpt-j-6b | pile_10k | lambada | channelx1  | 0.0999 | 0.0643 | 0.7917 | 0.7901 | -0.20% |
| facebook/opt-1.3b | pile_10k | lambada |  channelx1  | 0.0999 | 0.0614 | 0.7541 | 0.7498 | -0.57% |
| facebook/opt-2.7b | pile_10k | lambada |  channelx1  | 0.0999 | 0.0634 | 0.7779 | 0.7778 | -0.01% |
| decapoda-research/llama-7b-hf | pile_10k | lambada |  channelx1  | 0.0999 | 0.0654 | 0.8856 | 0.8815 | -0.46% |
| bigscience/bloom-1b7 | pile_10k | lambada |  channelx1  | 0.0999 | 0.0466 | 0.7143 | 0.7141 | -0.03% |
| bigscience/bloom-7b1 | pile_10k | lambada |  channelx1  | 0.0999 | 0.0568 | 0.7745 | 0.7742 | -0.04% |

<br />

The last word acc of the channel-wise sparse model is shown in the following table. All the sparsity is 10% over MLP block.
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
| EleutherAI/gpt-j-6b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6831 | 0.6911 | +1.17% |
| EleutherAI/gpt-j-6b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6792 | 0.6885 | +1.37% |
| decapoda-research/llama-7b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7361 | 0.7336 | -0.34% |
| decapoda-research/llama-7b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.7326 | 0.7316 | -0.14% |
| facebook/opt-6.7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6365 | 0.6621 | -2.19% |
| facebook/opt-6.7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6336 | 0.6586 | -2.23% |
| tiiuae/falcon-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7467 | 0.7415 | -0.70% |
| tiiuae/falcon-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.7471 | 0.7403 | -0.91% |
| bigscience/bloom-7b1 | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.5764 | 0.5575 | -3.28% |
| bigscience/bloom-7b1 | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.5723 | 0.5513 | -3.67% |
| decapoda-research/llama-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 50% | FP32 | 0.7627 | 0.7584 | -0.56% |
| decapoda-research/llama-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 50% | BF16 | 0.7601 | 0.7545 | -0.74% |

