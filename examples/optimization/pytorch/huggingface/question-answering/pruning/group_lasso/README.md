Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT pruning result.

# Prerequisite

## 1. Environment

Recommend python 3.7 or higher version.

### Install [intel-extension-for-transformers]()
```
pip install intel-extension-for-transformers
```

### Install PyTorch

Install pytorch-gpu, visit [pytorch.org](https://pytorch.org/).
```bash
# Install pytorch
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Install BERT dependency

```bash
cd examples/pytorch/huggingface/question-answering/pruning/group_lasso
pip3 install -r requirements.txt --ignore-installed PyYAML
```
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
> **Note**
>
> If no CUDA runtime is found, please export CUDA_HOME='/usr/local/cuda'.

## 2. Prepare Dataset

* For SQuAD task, you should download SQuAD dataset from [SQuAD dataset link](https://rajpurkar.github.io/SQuAD-explorer/).
## 3. Prepare Model
* Please download BERT large pretrained model from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_large_pretraining_amp_lamb/files?version=20.03.0).
```bash
# wget cmd
wget https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_large_pretraining_amp_lamb/versions/20.03.0/files/bert_large_pretrained_amp.pt

# curl cmd
curl -LO https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_large_pretraining_amp_lamb/versions/20.03.0/files/bert_large_pretrained_amp.pt
```
# Run
Enter your created conda env, then run the script.
```bash
bash scripts/run_squad_sparse.sh /path/to/model.pt 2.0 16 5e-5 tf32 /path/to/data /path/to/outdir prune_bert.yaml
```
The default parameters are as follows:
```shell
init_checkpoint=${1:-"/path/to/ckpt_8601.pt"}
epochs=${2:-"2.0"}
batch_size=${3:-"4"}
learning_rate=${4:-"3e-5"}
precision=${5:-"tf32"}
BERT_PREP_WORKING_DIR=${6:-'/path/to/bert_data'}
OUT_DIR=${7:-"./results/SQuAD"}
prune_config=${8:-"prune_bert.yaml"}
```
 >**Note**: For original BERT readme, please refer [BERT README](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/README.md)
