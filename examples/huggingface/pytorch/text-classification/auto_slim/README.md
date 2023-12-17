# Step by Step

## Channel Pruning for Consecutive Linear Layers
An interesting thing for pruning is that if we do [channel pruning](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner#pruning-patterns) for some linear layers in NLP models, we can permanently remove these all-zero channels without changing their accuracy. 

To be specific, if a model has two consecutive linear layers, which is common in both **Bert series** and **GPT series** models' FFN parts, and we conduct the input channel pruning for the second linear layer (masking weights by column). We can remove these all-zero channels. Plus, we also remove the same indices' output channels in the first linear layers (masking weights by row), since their contribution for activation will be masked by the second layer's. 

This leads to no change for model's accuracy, but can obtain a significant acceleration for model's inference, because the transformer models' FFN parts take nearly 50% of entire computing overhead. Thus, compressing weights in FFN parts is really useful.

## Multi-head Pruning for Self-Attention Layers
Self attention modules are common in all Transformer-based models. These models use multi-head attention (also known as MHA) to enhance their abilities of linking contextual information. Transformer-based models usually stack a sequence of MHA modules, and this makes MHA takes a noticeable storage and memory bandwidth. As an optimization method, head pruning removes attention heads which make minor contribution to model's contextual analysis. This method does not lead to much accuracy loss, but provides us with much opportunity for model acceleration. 

## API for Consecutive Linear Layers and Multi-head attention Slim.
We provide API functions for you to complete the process above and slim your transformer models easily. Here is how to call our API functions. Simply provide a target sparsity value to our Our API function **parse_auto_slim_config** and it can generate the [pruning_configs](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner#get-started-with-pruning-api) used by our pruning API. Such process is fully automatic and target linear layers will be included without manual setting. After pruning process finished, use API function **model_slim** to slim the model.

```python
# auto slim config
# part1 generate pruning configs for the second linear layers. 
pruning_configs = []
from neural_compressor.compression.pruner import parse_auto_slim_config
auto_slim_configs = parse_auto_slim_config(
    model, 
    ffn2_sparsity = prune_ffn2_sparsity, # define target sparsity with a float between 0 and 1
    mha_sparsity = prune_mha_sparsity, # define target sparsity with a float between 0 and 1
)
pruning_configs += auto_slim_configs

################
"""
# Training codes.
......
"""
################

from neural_compressor.compression.pruner import model_slim
model = model_slim(model)
```
Please noted that if you already have a sparse model which corresponding linear layers pruned, you can simply call the last two lines to complete the model slim. 

## Run Examples on BGE Models
For an introduction for BGE models, please refer to following [link](https://huggingface.co/BAAI/bge-small-en-v1.5). \
We provides an example of [bge-small](https://huggingface.co/BAAI/bge-small-en-v1.5) and [bge-base](https://huggingface.co/BAAI/bge-base-en-v1.5) models to demonstrate how we slim Transformer-based models. In this example, we simultaneously prune the searched feed forward networks and multi-head attention modules to obtain the best acceleration performance. Simply run the following script:
```bash
sh run_auto_slim.sh
```
Here are some results of our auto slim performance on BGE models, trained on [MTEB/stsbenchmark-sts](https://huggingface.co/datasets/mteb/stsbenchmark-sts). According to the results, we can slim around original models' 50% of parameters, while keeping a marginal accuracy loss.

| Model    | Finetuned pearson/spearanr | After auto-slim pearson/spearanr | Relative loss pearson | Model parameters before/after auto slim | Slim ratio |
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| bge-small-en-v1.5 | 0.865380 / 0.871302 |0.861265 / 0.867966 | -0.48% / -0.38% | 33360385 / 18002305 | 46.04% |
| bge-base-en-v1.5 | 0.872808 / 0.873081 | 0.870472 / 0.867611 | -0.27% / -0.63% | 109483009 / 48096001  | 56.07% |
