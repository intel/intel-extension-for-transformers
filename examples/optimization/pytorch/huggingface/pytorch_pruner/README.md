# Pytorch Pruner

## Introdunction
**Pytorch Pruner** is an build-in API which supports a wide range of pruning algorithms, patterns as well as pruning schedulers. Features below are currently supported:
> algorithms: magnitude, snip, snip-momentum\
> patterns: NxM, N:M\
> pruning schedulers: iterative pruning scheduler, oneshot pruning scheduler.

## Getting Started
### Prepare Config File
Pytorch pruner is developed based on pruning, therefore most usages are identical. Our API reads in a yaml configuration file to define a Pruning object. Here is an bert-mini example of it:
```yaml
model:
  name: "bert-mini"
  framework: "pytorch"

pruning:
  approach:
    weight_compression_pytorch:
      # Global settings
      # if start step equals to end step, oneshot pruning scheduler is enabled. Otherwise the API automatically implements iterative pruning scheduler.
      start_step: 0 # step which pruning process begins
      end_step: 0 # step which pruning process ends
      not_to_prune_names: ["classifier", "pooler", ".*embeddings*"] # a global announcement of layers which you do not wish to prune. 
      prune_layer_type: ["Linear"] # the module type which you want to prune (Linear, Conv2d, etc.)
      target_sparsity: 0.9 # the sparsity you want the model to be pruned.
      max_sparsity_ratio_per_layer: 0.98 # the sparsity ratio's maximum which one layer can reach.

      pruners: # below each "Pruner" defines a pruning process for a group of layers. This enables us to apply different pruning methods for different layers in one model.
        # Local settings
        - !Pruner
            exclude_names: [".*query", ".*key", ".*value"] # list of regular expressions, containing the layer names you wish not to be included in this pruner
            pattern: "1x1" # pattern type, we support "NxM" and "N:M"
            update_frequency_on_step: 100 # if use iterative pruning scheduler, this define the pruning frequency.
            prune_domain: "global" # one in ["global", "local"], refers to the score map is computed out of entire parameters or its corresponding layer's weight.
            prune_type: "snip_momentum" # pruning algorithms, refer to pytorch_pruner/pruner.py
            sparsity_decay_type: "exp" # ["linear", "cos", "exp", "cube"] ways to determine the target sparsity during iterative pruning.
        - !Pruner
            exclude_names: [".*output", ".*intermediate"]
            pattern: "4x1"
            update_frequency_on_step: 100
            prune_domain: "global"
            prune_type: "snip_momentum"
            sparsity_decay_type: "exp"
```
Please be awared that when the keywords appear in both global and local settings, we select the **local** settings as priority.

### Quick Sample:
With a settled config file, we provide a template for implementing pytorch_pruner API:
```python
model = Model()
criterion = Criterion()
optimizer = Optimizer()
args = Args()

from neural_compressor.experimental.pytorch_pruner.pruning import Pruning

pruner = Pruning("path/to/your/config.yaml")
if args.do_prune:
    pruner.update_items_for_all_pruners(start_step=int(args.sparsity_warm_epochs * num_iterations), end_step=int(total_iterations))  ##iterative
else:
   pruner.update_items_for_all_pruners(start_step=total_iterations+1, end_step=total_iterations+1) ## remove the pruner
pruner.model = model
pruner.on_train_begin()
for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        pruner.on_step_begin(step)
        output = model(**batch)
        loss = output.loss
        loss.backward()
        pruner.on_before_optimizer_step()
        optimizer.step()
        pruner.on_after_optimizer_step()
        optimizer.zero_grad()
    
    model.eval()
    for step, batch in enumerate(val_dataloader):
        ...
```
For more usage, please refer to our example codes below.

## Step-by-Step Examples
### Installation
Install Intel® Extension for Transformers, please refer to [installation](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/installation.md)
```shell
# Install from pypi
pip install intel-extension-for-transformers
```
Install required dependencies for examples
```shell
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/squad/bert_large
pip install -r requirements.txt
```

### Pruning
We have provided several pruning examples, which are trained on different datasets/tasks, use different sparsity patterns, etc. 

We can train a sparse model with NxM (2:4) pattern on mrpc and sst2:
```
python3 ./run_glue_no_trainer.py \
            --model_name_or_path "prajjwal1/bert-mini" \
            --pruning_config "./bert_mini_mrpc_2in4.yaml" \
            --task_name "mrpc" \
            --max_length "128" \
            --per_device_train_batch_size "16" \
            --learning_rate 5e-5 \
            --num_train_epochs 10 \
            --weight_decay 5e-5   \
            --lr_scheduler_type "constant" \
	    --seed 9 \
	    --sparsity_warm_epochs 1 \
	    --cooldown_epochs 0 \
	    --do_prune \
            --output_dir "pruned_mrpc_bert-mini"
```
```
python ./run_glue_no_trainer.py \
        --model_name_or_path "prajjwal1/bert-mini" \
        --pruning_config "./bert_mini_sst2_2in4.yaml" \
            --task_name "sst2" \
            --max_length "128" \
            --per_device_train_batch_size "16" \
            --learning_rate 5e-5 \
	    --weight_decay 1e-4 \
            --num_train_epochs 6 \
            --sparsity_warm_epochs 0 \
	    --seed 12 \
            --output_dir "pruned_sst2_bert-mini"
```
We can also choose a NxM (4x1) pattern:
```
python3 ./run_glue_no_trainer.py \
        --model_name_or_path "prajjwal1/bert-mini" \
        --pruning_config "./bert_mini_mrpc_4x1.yaml" \
        --task_name "mrpc" \
        --max_length "128" \
        --per_device_train_batch_size "16" \
        --learning_rate 1e-3 \
        --num_train_epochs 15 \
        --weight_decay 1e-3  \
        --cooldown_epochs 5 \
        --sparsity_warm_epochs 1 \
        --lr_scheduler_type "constant"\
        --distill_loss_weight 5 \
        --do_prune \
        --output_dir "pruned_mrpc_bert-mini"
```
```
python3 ./run_glue_no_trainer.py \
        --model_name_or_path ".prajjwal1/bert-mini" \
        --pruning_config "./bert_mini_sst2_4x1.yaml" \
        --task_name "sst2" \
        --max_length "128" \
        --per_device_train_batch_size "16" \
        --learning_rate 5e-5 \
        --distill_loss_weight 2.0 \
        --num_train_epochs 15 \
        --weight_decay 5e-5   \
        --cooldown_epochs 5 \
        --sparsity_warm_epochs 0 \
        --lr_scheduler_type "constant" \
        --do_prune \
        --output_dir "pruned_sst2_bert-mini"
```
Dense model traing on glue datasets is supportted as well, by setting `--do_prune` to False, for example on sst2:
```
python run_glue_no_trainer.py --model_name_or_path ./bert-mini --task_name sst2 --max_length 128 --per_device_train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 10  --output_dir result/ 2>&1 | tee  sst2_orig.log
```
or on mrpc,
```
python3 run_glue_no_trainer.py  --model_name_or_path ./bert-mini  --task_name mrpc --max_length 128 --per_device_train_batch_size 16  --learning_rate 5e-5 --num_train_epoch 5 --weight_decay 5e-5 --output_dir result/ 2>&1 | tee sst2_snip.log 
```
## Results
### MRPC
|  Model  | Dataset  | Sparsity pattern | Pruning methods |Element-wise/matmul, Gemm, conv ratio | Init model | Dense F1 (mean/max) | Sparse F1 (mean/max) | Relative drop |
|  :----:  | :----:  | :----: | :----: |:----:|:----:| :----: | :----: | :----: |
| Bert-Mini  | MRPC |  4x1  |Snip-momentum| 0.8804 | Dense & Finetuned | 0.8619/0.8752 | 0.8610/0.8722 | -0.34% |
| Bert-Mini  | MRPC |  2:4  |Snip-momentum| 0.4795 | Dense & Finetuned | 0.8619/0.8752| 0.8562/0.8695 | -0.65% |

### SST-2
|  Model  | Dataset  |  Sparsity pattern | Pruning methods |Element-wise/matmul, Gemm, conv ratio | Init model | Dense Accuracy (mean/max) | Sparse Accuracy (mean/max)| Relative drop|
|  :----:  | :----:  | :----: | :----: |:----:|:----:| :----: | :----: | :----: |
| Bert-Mini  | SST-2 |  4x1  |Snip-momentum| 0.8815 | Dense & Finetuned | 0.8660/0.8761 | 0.8651/0.8692 | -0.79% |
| Bert-Mini  | SST-2 |  2:4  |Snip-momentum| 0.4795 | Dense & Finetuned | 0.8660/0.8761 | 0.8609/0.8693| -0.78% |

## References
* [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
* [Knowledge Distillation with the Reused Teacher Classifier](https://arxiv.org/abs/2203.14001)
