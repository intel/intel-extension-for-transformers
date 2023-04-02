Step-by-Step
============

This document is used to list steps of reproducing PyTorch longformer-base-4096 pruning result.


# Prerequisite

## 1. Environment


```shell
pip install -r requirements.txt
```

## 2. Prepare Dataset

The dataset will be downloaded and converted to squad format automatically with `./scripts/download_data_and_convert.sh`.

```shell
bash ./scripts/download_data_and_convert.sh
```

There will generate two squad format files: `squad-wikipedia-train-4096.json` and `squad-wikipedia-dev-4096.json`


# Run Examples

### pruning longformer-base-4096

Run the `./scripts/longformer_base_sparse_global_4x1_pruning.sh` to prune with `global sparse 80% and 4*1 pattern`. In this script, we set `per_device_train_batch_size=1` which is same with [the original longformer codes](https://github.com/allenai/longformer).

```shell
bash ./scripts/longformer_base_sparse_global_4x1_pruning.sh
```

Fine-tuning of the dense model is also supported by running the `./scripts/longformer_base_dense_fintune.sh`


### Results
The snip-momentum pruning method is used by default and the initial dense model is well fine-tuned.

|  Model  | Dataset  |  Sparsity pattern | sparsity ratio | Dense F1  |Sparse F1 | Relative drop|
|  :----:  | :----:  | :----: | :----: |:----: |:----:| :----: |
| longformer-base-4096 | triviaqa |  4x1  | global 80% | 75.2 (from [the paper](https://arxiv.org/abs/2004.05150))/74.9235 (ours) | 74.48 | -0.96% |

## References
* [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

