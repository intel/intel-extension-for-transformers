# Run Using Docker

## Build the Container

```bash
docker compose build dev
```

## Run the Container

```bash
$ CONFIG=<config_name> docker compose run dev
```

### Example

```bash
$ CONFIG=qat docker compose run dev
```

### Expected Output

```
$ CONFIG=qat docker compose run dev
[+] Running 1/0
 ⠿ Network docker_default  Created                                                             0.0s
[+] Running 0/1
 ⠿ dev Error                                                                                   1.0s
[+] Building 89.8s (11/11) FINISHED                                                                 
 => [internal] load build definition from Dockerfile                                           0.0s
 => => transferring dockerfile: 411B                                                           0.0s
 => [internal] load .dockerignore                                                              0.0s
 => => transferring context: 2B                                                                0.0s
 => [internal] load metadata for docker.io/intel/intel-optimized-pytorch:pip-ipex-1.13.100-ub  0.3s
 => CACHED [1/6] FROM docker.io/intel/intel-optimized-pytorch:pip-ipex-1.13.100-ubuntu-22.04@  0.0s
 => [internal] load build context                                                              0.0s
 => => transferring context: 10.76kB                                                           0.0s
 => [2/6] RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing      26.3s
 => [3/6] RUN mkdir -p /workspace/output                                                       0.5s
 => [4/6] COPY . /workspace                                                                    0.0s 
 => [5/6] WORKDIR /workspace                                                                   0.0s 
 => [6/6] RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt             58.4s 
 => exporting to image                                                                         4.2s 
 => => exporting layers                                                                        4.2s 
 => => writing image sha256:c861ce70310454071a6abe3b9fd72319998ef2fcc59562ada9762106e5f512e2   0.0s 
 => => naming to docker.io/intel/ai-workflows:beta-compression-aware                           0.0s 
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).  
04/12/2023 17:48:10 - WARNING - itrex_opt - Process rank: -1, device: cpu, n_gpu: 0distributed training: False, 16-bits training: False
04/12/2023 17:48:10 - INFO - itrex_opt - Training/evaluation parameters TrainingArguments(
_n_gpu=0,
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_pin_memory=True,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
do_eval=True,
do_predict=False,
do_train=True,
eval_accumulation_steps=None,
eval_delay=0,
eval_steps=None,
evaluation_strategy=no,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_model_id=None,
hub_private_repo=False,
hub_strategy=every_save,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_inputs_for_metrics=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=5e-05,
length_column_name=length,
load_best_model_at_end=False,
local_rank=-1,
log_level=passive,
log_level_replica=passive,
log_on_each_node=True,
logging_dir=./output/runs/Apr12_17-48-10_f1d33e4bf92f,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=500,
logging_strategy=steps,
lr_scheduler_type=linear,
max_grad_norm=1.0,
max_steps=-1,
metric_for_best_model=None,
mp_parameters=,
no_cuda=False,
num_train_epochs=3.0,
optim=adamw_hf,
optim_args=None,
output_dir=./output,
overwrite_output_dir=True,
past_index=-1,
per_device_eval_batch_size=8,
per_device_train_batch_size=8,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=[],
resume_from_checkpoint=None,
run_name=./output,
save_on_each_node=False,
save_steps=500,
save_strategy=steps,
save_total_limit=None,
seed=42,
sharded_ddp=[],
skip_memory_metrics=True,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_ipex=False,
use_legacy_prediction_loop=False,
use_mps_device=False,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.0,
xpu_backend=None,
)
04/12/2023 17:48:11 - INFO - datasets.utils.file_utils - https://huggingface.co/datasets/SetFit/emotion/resolve/main/README.md not found in cache or force_download set to True, downloading to /root/.cache/huggingface/datasets/downloads/tmpvs5409da
Downloading readme: 100%|███████████████████████████████████████████| 194/194 [00:00<00:00, 330kB/s]
04/12/2023 17:48:11 - INFO - datasets.utils.file_utils - storing https://huggingface.co/datasets/SetFit/emotion/resolve/main/README.md in cache at /root/.cache/huggingface/datasets/downloads/cd1b5cefb87f4927158378fe736baec563c4910e3f39d3d10c0c812ef780a23d.dd261a831cbf45e5c6bb8199db90b7a619caa8a6a8de7dbdd1eb49bd51b5f2d9
04/12/2023 17:48:11 - INFO - datasets.utils.file_utils - creating metadata file for /root/.cache/huggingface/datasets/downloads/cd1b5cefb87f4927158378fe736baec563c4910e3f39d3d10c0c812ef780a23d.dd261a831cbf45e5c6bb8199db90b7a619caa8a6a8de7dbdd1eb49bd51b5f2d9
04/12/2023 17:48:11 - INFO - datasets.builder - Using custom data configuration SetFit--emotion-e444b7640ce3116e
04/12/2023 17:48:11 - INFO - datasets.info - Loading Dataset Infos from /usr/local/lib/python3.10/dist-packages/datasets/packaged_modules/json
04/12/2023 17:48:11 - INFO - datasets.builder - Generating dataset json (/root/.cache/huggingface/datasets/SetFit___json/SetFit--emotion-e444b7640ce3116e/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e)
Downloading and preparing dataset json/SetFit--emotion to /root/.cache/huggingface/datasets/SetFit___json/SetFit--emotion-e444b7640ce3116e/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e...
04/12/2023 17:48:11 - INFO - datasets.builder - Dataset not on Hf google storage. Downloading and preparing it from source
Downloading data files:   0%|                                                 | 0/3 [00:00<?, ?it/s]04/12/2023 17:48:12 - INFO - datasets.utils.file_utils - https://huggingface.co/datasets/SetFit/emotion/resolve/6c362e04d016f6b6a9377e85c3b944140f0b96c9/train.jsonl not found in cache or force_download set to True, downloading to /root/.cache/huggingface/datasets/downloads/tmpserk2lk1
Downloading data: 100%|████████████████████████████████████████| 2.23M/2.23M [00:00<00:00, 27.4MB/s]
04/12/2023 17:48:12 - INFO - datasets.utils.file_utils - storing https://huggingface.co/datasets/SetFit/emotion/resolve/6c362e04d016f6b6a9377e85c3b944140f0b96c9/train.jsonl in cache at /root/.cache/huggingface/datasets/downloads/f92bd534c53edf25a9cf5dfc34a00f0a0052d796a5076e1f6393ee24c2ecaa9e
04/12/2023 17:48:12 - INFO - datasets.utils.file_utils - creating metadata file for /root/.cache/huggingface/datasets/downloads/f92bd534c53edf25a9cf5dfc34a00f0a0052d796a5076e1f6393ee24c2ecaa9e
Downloading data files:  33%|█████████████▋                           | 1/3 [00:00<00:01,  1.49it/s]04/12/2023 17:48:12 - INFO - datasets.utils.file_utils - https://huggingface.co/datasets/SetFit/emotion/resolve/6c362e04d016f6b6a9377e85c3b944140f0b96c9/test.jsonl not found in cache or force_download set to True, downloading to /root/.cache/huggingface/datasets/downloads/tmp_52j2l4o
Downloading data: 100%|██████████████████████████████████████████| 279k/279k [00:00<00:00, 19.3MB/s]
04/12/2023 17:48:13 - INFO - datasets.utils.file_utils - storing https://huggingface.co/datasets/SetFit/emotion/resolve/6c362e04d016f6b6a9377e85c3b944140f0b96c9/test.jsonl in cache at /root/.cache/huggingface/datasets/downloads/f3b58e9758db1338bdda3704fc34dc5c9490651613e0dee0caed98992608522a
04/12/2023 17:48:13 - INFO - datasets.utils.file_utils - creating metadata file for /root/.cache/huggingface/datasets/downloads/f3b58e9758db1338bdda3704fc34dc5c9490651613e0dee0caed98992608522a
Downloading data files:  67%|███████████████████████████▎             | 2/3 [00:01<00:00,  1.73it/s]04/12/2023 17:48:13 - INFO - datasets.utils.file_utils - https://huggingface.co/datasets/SetFit/emotion/resolve/6c362e04d016f6b6a9377e85c3b944140f0b96c9/validation.jsonl not found in cache or force_download set to True, downloading to /root/.cache/huggingface/datasets/downloads/tmpohr11os7
Downloading data: 100%|██████████████████████████████████████████| 276k/276k [00:00<00:00, 18.0MB/s]
04/12/2023 17:48:13 - INFO - datasets.utils.file_utils - storing https://huggingface.co/datasets/SetFit/emotion/resolve/6c362e04d016f6b6a9377e85c3b944140f0b96c9/validation.jsonl in cache at /root/.cache/huggingface/datasets/downloads/b493840d4bba52a0d3ebf9501d04c31412a7fd89a7e14e442d4cbe9c50f9e2d4
04/12/2023 17:48:13 - INFO - datasets.utils.file_utils - creating metadata file for /root/.cache/huggingface/datasets/downloads/b493840d4bba52a0d3ebf9501d04c31412a7fd89a7e14e442d4cbe9c50f9e2d4
Downloading data files: 100%|█████████████████████████████████████████| 3/3 [00:01<00:00,  1.77it/s]
04/12/2023 17:48:13 - INFO - datasets.download.download_manager - Downloading took 0.0 min
04/12/2023 17:48:13 - INFO - datasets.download.download_manager - Checksum Computation took 0.0 min
Extracting data files: 100%|████████████████████████████████████████| 3/3 [00:00<00:00, 2548.70it/s]
04/12/2023 17:48:13 - INFO - datasets.builder - Generating train split
04/12/2023 17:48:13 - INFO - datasets.builder - Generating test split
04/12/2023 17:48:13 - INFO - datasets.builder - Generating validation split
04/12/2023 17:48:13 - INFO - datasets.utils.info_utils - Unable to verify splits sizes.
Dataset json downloaded and prepared to /root/.cache/huggingface/datasets/SetFit___json/SetFit--emotion-e444b7640ce3116e/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e. Subsequent calls will reuse this data.
100%|███████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 1045.53it/s]
Step 1: Load the emotion dataset
################################
```
...
```
Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 1322.208, 'train_samples_per_second': 36.303, 'train_steps_per_second': 4.538, 'train_loss': 0.3860068677266439, 'epoch': 3.0}
100%|███████████████████████████████████████████████████████████| 6000/6000 [22:02<00:00,  4.54it/s]
2023-04-12 18:10:25 [INFO] Saving model checkpoint to ./output
[INFO|configuration_utils.py:453] 2023-04-12 18:10:25,021 >> Configuration saved in ./output/config.json
[INFO|modeling_utils.py:1704] 2023-04-12 18:10:25,287 >> Model weights saved in ./output/pytorch_model.bin
[INFO|tokenization_utils_base.py:2160] 2023-04-12 18:10:25,299 >> tokenizer config file saved in ./output/tokenizer_config.json
[INFO|tokenization_utils_base.py:2167] 2023-04-12 18:10:25,299 >> Special tokens file saved in ./output/special_tokens_map.json
***** train metrics *****
  epoch                    =        3.0
  train_loss               =      0.386
  train_runtime            = 0:22:02.20
  train_samples_per_second =     36.303
  train_steps_per_second   =      4.538
2023-04-12 18:10:25 [INFO] |*********Mixed Precision Statistics********|
2023-04-12 18:10:25 [INFO] +---------------------+-------+------+------+
2023-04-12 18:10:25 [INFO] |       Op Type       | Total | INT8 | FP32 |
2023-04-12 18:10:25 [INFO] +---------------------+-------+------+------+
2023-04-12 18:10:25 [INFO] |      Embedding      |   3   |  3   |  0   |
2023-04-12 18:10:25 [INFO] |      LayerNorm      |   9   |  0   |  9   |
2023-04-12 18:10:25 [INFO] | quantize_per_tensor |   26  |  26  |  0   |
2023-04-12 18:10:25 [INFO] |        Linear       |   26  |  26  |  0   |
2023-04-12 18:10:25 [INFO] |      dequantize     |   26  |  26  |  0   |
2023-04-12 18:10:25 [INFO] |     input_tensor    |   8   |  8   |  0   |
2023-04-12 18:10:25 [INFO] |       Dropout       |   8   |  0   |  8   |
2023-04-12 18:10:25 [INFO] +---------------------+-------+------+------+
2023-04-12 18:10:25 [INFO] Pass quantize model elapsed time: 1323342.72 ms
2023-04-12 18:10:25 [INFO] The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: label_text, text.
[INFO|trainer.py:2964] 2023-04-12 18:10:25,728 >> ***** Running Evaluation *****
[INFO|trainer.py:2966] 2023-04-12 18:10:25,728 >>   Num examples = 2000
[INFO|trainer.py:2969] 2023-04-12 18:10:25,728 >>   Batch size = 8
 98%|███████████████████████████████████████████████████████████▊ | 245/250 [00:03<00:00, 67.95it/s]04/12/2023 18:10:29 - INFO - datasets.metric - Removing /root/.cache/huggingface/metrics/accuracy/default/default_experiment-1-0.arrow
100%|█████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 71.47it/s]
2023-04-12 18:10:29 [INFO] {
2023-04-12 18:10:29 [INFO]     'eval_loss': 0.21442773938179016,
2023-04-12 18:10:29 [INFO]     'eval_accuracy': 0.93,
2023-04-12 18:10:29 [INFO]     'eval_runtime': 3.5182,
2023-04-12 18:10:29 [INFO]     'eval_samples_per_second': 568.471,
2023-04-12 18:10:29 [INFO]     'eval_steps_per_second': 71.059,
2023-04-12 18:10:29 [INFO]     'epoch': 3.0
2023-04-12 18:10:29 [INFO] }
2023-04-12 18:10:29 [INFO] metric: 0.93
2023-04-12 18:10:29 [INFO] Throughput: 568.471 samples/sec
2023-04-12 18:10:29 [INFO] Tune 1 result is: [Accuracy (int8|fp32): 0.9300|0.2750, Duration (seconds) (int8|fp32): 3.5210|3.7284], Best tune result is: [Accuracy: 0.9300, Duration (seconds): 3.5210]
2023-04-12 18:10:29 [INFO] |**********************Tune Result Statistics**********************|
2023-04-12 18:10:29 [INFO] +--------------------+----------+---------------+------------------+
2023-04-12 18:10:29 [INFO] |     Info Type      | Baseline | Tune 1 result | Best tune result |
2023-04-12 18:10:29 [INFO] +--------------------+----------+---------------+------------------+
2023-04-12 18:10:29 [INFO] |      Accuracy      | 0.2750   |    0.9300     |     0.9300       |
2023-04-12 18:10:29 [INFO] | Duration (seconds) | 3.7284   |    3.5210     |     3.5210       |
2023-04-12 18:10:29 [INFO] +--------------------+----------+---------------+------------------+
2023-04-12 18:10:29 [INFO] Save tuning history to /workspace/nc_workspace/2023-04-12_17-48-08/./history.snapshot.
2023-04-12 18:10:29 [INFO] Specified timeout or max trials is reached! Found a quantized model which meet accuracy goal. Exit.
2023-04-12 18:10:29 [INFO] Save deploy yaml to /workspace/nc_workspace/2023-04-12_17-48-08/deploy.yaml
2023-04-12 18:10:29 [INFO] Saving model checkpoint to ./output
[INFO|configuration_utils.py:453] 2023-04-12 18:10:29,256 >> Configuration saved in ./output/config.json
2023-04-12 18:10:29 [INFO] quantized model and configure file have saved to ./output
[INFO|tokenization_utils_base.py:2160] 2023-04-12 18:10:29,311 >> tokenizer config file saved in ./output/tokenizer_config.json
[INFO|tokenization_utils_base.py:2167] 2023-04-12 18:10:29,311 >> Special tokens file saved in ./output/special_tokens_map.json
```

### Cleanup

```bash
$ docker compose down
```
