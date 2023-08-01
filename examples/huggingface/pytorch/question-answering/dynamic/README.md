Step-by-step
============

Quantized Length Adaptive Transformer is based on [Length Adaptive Transformer](https://github.com/clovaai/length-adaptive-transformer)'s work. Currently, it supports BERT and Reberta based transformers.

[QuaLA-MiniLM: A Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114) has been accepted by NeurIPS 2022. Our quantized length-adaptive MiniLM model (QuaLA-MiniLM) is trained only once, dynamically fits any inference scenario, and achieves an accuracy-efficiency trade-off superior to any other efficient approaches per any computational budget on the SQuAD1.1 dataset (up to x8.8 speedup with <1% accuracy loss). The following shows how to reproduce this work and we also provide the [jupyter notebook tutorials](../../../../../../docs/tutorials/pytorch/question-answering/Dynamic_MiniLM_SQuAD.ipynb).

# Prerequisite​

## 1. Environment
```
pip install intel-extension-for-transformers
pip install -r requirements.txt
```
  >*Note: Suggest use PyTorch 1.12.0 and Intel Extension for PyTorch 1.12.0

# Run


## Step 1: Finetune
In this step, `output/finetuning` is a fine-tuned minilm for squad, which uploaded to [sguskin/minilmv2-L6-H384-squad1.1](https://huggingface.co/sguskin/minilmv2-L6-H384-squad1.1)
```
python run_qa.py \
--model_name_or_path nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large \
--dataset_name squad \
--do_train \
--do_eval \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--per_device_train_batch_size 8 \
--output_dir output/finetuning
```


## Step 2: Training with LengthDrop
Train it with length-adaptive training to get the dynamic model `output/finetuning` which uploaded to [sguskin/dynamic-minilmv2-L6-H384-squad1.1](https://huggingface.co/sguskin/dynamic-minilmv2-L6-H384-squad1.1)

```
python run_qa.py \
--model_name_or_path output/finetuning \
--dataset_name squad \
--do_train \
--do_eval \
--learning_rate 3e-5 \
--num_train_epochs 5 \
--max_seq_length 384 \
--doc_stride 128 \
--per_device_train_batch_size 8 \
--length_adaptive \
--num_sandwich 2  \
--length_drop_ratio_bound 0.2 \
--layer_dropout_prob 0.2 \
--output_dir output/dynamic 

```


## Step 3: Evolutionary Search

Run evolutionary search to optimize length configurations for any possible target computational budget.

```
python run_qa.py \
--model_name_or_path output/dynamic \
--dataset_name squad \
--max_seq_length 384 \
--doc_stride 128 \
--do_eval \
--per_device_eval_batch_size 32 \
--do_search \
--output_dir output/search

```

## Step 4: Quantization

```
python run_qa.py \
--model_name_or_path "sguskin/dynamic-minilmv2-L6-H384-squad1.1" \
--dataset_name squad \
--quantization_approach PostTrainingStatic \
--do_eval \
--do_train \
--tune \
--output_dir output/quantized-dynamic-minilmv \
--overwrite_cache \
--per_device_eval_batch_size 32 \
--overwrite_output_dir
```


## Step 5: Apply Length Config for Quantization
```
python run_qa.py \
--model_name_or_path "sguskin/dynamic-minilmv2-L6-H384-squad1.1" \  # used for load int8 model.
--dataset_name squad \
--do_eval \
--accuracy_only \
--int8 \
--output_dir output/quantized-dynamic-minilmv \  # used for load int8 model
--overwrite_cache \
--per_device_eval_batch_size 32 \
--length_config "(315, 251, 242, 159, 142, 33)"
```


# Performance Data
Performance results test on ​​07/10/2022 with Intel Xeon Platinum 8280 Scalable processor, batchsize = 32
Performance varies by use, configuration and other factors. See platform configuration for configuration details. For more complete information about performance and benchmark results, visit www.intel.com/benchmarks


<table>
<thead>
  <tr>
    <th rowspan="2"><br>Model Name</th>
    <th rowspan="2">Datatype</th>
    <th rowspan="2"><br>Optimization Method</th>
    <th rowspan="2"><br><br><br>Modelsize (MB)</th>
    <th colspan="4"><br>InferenceResult</th>
  </tr>
  <tr>
    <th><br>Accuracy(F1)</th>
    <th><br>Latency(ms)</th>
    <th><br>GFLOPS**</th>
    <th><br>Speedup<br><br>(comparedwith BERT Base)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><br>BERT Base</td>
    <td>fp32</td>
    <td><br>None</td>
    <td><br>415.47</td>
    <td><br>88.58</td>
    <td><br>56.56</td>
    <td><br>35.3</td>
    <td><br>1x</td>
  </tr>
  <tr>
    <td><br>TinyBERT</td>
    <td>fp32</td>
    <td><br>Distillation</td>
    <td><br>253.20</td>
    <td><br>88.39</td>
    <td><br>32.40</td>
    <td><br>17.7</td>
    <td><br>1.75x</td>
  </tr>
  <tr>
    <td><br>QuaTinyBERT</td>
    <td>int8</td>
    <td><br>Distillation + quantization</td>
    <td><br>132.06</td>
    <td><br>87.67</td>
    <td><br>15.58</td>
    <td><br>17.7</td>
    <td><br>3.63x</td>
  </tr>
  <tr>
    <td><br>MiniLMv2</td>
    <td>fp32</td>
    <td><br>Distillation</td>
    <td><br>115.04</td>
    <td><br>88.70</td>
    <td><br>18.23</td>
    <td><br>4.76</td>
    <td><br>3.10x</td>
  </tr>
  <tr>
    <td><br>QuaMiniLMv2</td>
    <td>int8</td>
    <td><br>Distillation + quantization</td>
    <td><br>84.85</td>
    <td><br>88.54</td>
    <td><br>9.14</td>
    <td><br>4.76</td>
    <td><br>6.18x</td>
  </tr>
  <tr>
    <td><br>LA-MiniLM</td>
    <td>fp32</td>
    <td><br>Drop and restore base MiniLMv2</td>
    <td><br>115.04</td>
    <td><br>89.28</td>
    <td><br>16.99</td>
    <td><br>4.76</td>
    <td><br>3.33x</td>
  </tr>
  <tr>
    <td><br>LA-MiniLM(269, 253, 252, 202, 104, 34)*</td>
    <td>fp32</td>
    <td><br>Evolution search (best config)</td>
    <td><br>115.04</td>
    <td><br>87.76</td>
    <td><br>11.44</td>
    <td><br>2.49</td>
    <td><br>4.94x</td>
  </tr>
  <tr>
    <td><br>QuaLA-MiniLM</td>
    <td>int8</td>
    <td><br>Quantization base LA-MiniLM</td>
    <td><br>84.85</td>
    <td><br>88.85</td>
    <td><br>7.84</td>
    <td><br>4.76</td>
    <td><br>7.21x</td>
  </tr>
  <tr>
    <td><br>QuaLA-MiniLM(315,251,242,159,142,33)*</td>
    <td>int8</td>
    <td><br>Evolution search (best config)</td>
    <td><br>84.86</td>
    <td><br>87.68</td>
    <td><br>6.41</td>
    <td><br>2.55</td>
    <td><br>8.82x</td>
  </tr>
</tbody>
</table>
NOTES: * length config apply to LA model


NOTES: ** the multiplication and addition operation amount when model inference  (GFLOPS is obtained from torchprofile tool)


# Platform Configuration

<table>
<tbody>
  <tr>
    <td>Manufacturer</td>
    <td>Intel Corporation</td>
  </tr>
  <tr>
    <td>Product Name</td>
    <td>S2600WFD</td>
  </tr>
  <tr>
    <td>BIOS Version</td>
    <td>1SE5C620.86B.02.01.0008.031920191559</td>
  </tr>
  <tr>
    <td>OS</td>
    <td>CentOS Linux release 8.4.2105</td>
  </tr>
  <tr>
    <td>Kernel</td>
    <td>4.18.0-305.3.1.el8.x86_64</td>
  </tr>
  <tr>
    <td>Microcode</td>
    <td>0x5003006</td>
  </tr>
  <tr>
    <td>IRQ Balance</td>
    <td>Eabled</td>
  </tr>
  <tr>
    <td>CPU Model</td>
    <td>Intel(R) Xeon Platinum 8280 CPU @ 2.70GHz</td>
  </tr>
  <tr>
    <td>Base Frequency</td>
    <td>2.7GHz</td>
  </tr>
  <tr>
    <td>Maximum Frequency</td>
    <td>4.0GHz</td>
  </tr>
  <tr>
    <td>All-core Maximum Frequency</td>
    <td>3.3GHz</td>
  </tr>
  <tr>
    <td>CPU(s)</td>
    <td>112</td>
  </tr>
  <tr>
    <td>Thread(s) per Core</td>
    <td>2</td>
  </tr>
  <tr>
    <td>Core(s) per Socket</td>
    <td>28</td>
  </tr>
  <tr>
    <td>Socket(s)</td>
    <td>2</td>
  </tr>
  <tr>
    <td>NUMA Node(s)</td>
    <td>2</td>
  </tr>
  <tr>
    <td>Turbo</td>
    <td>Enabled</td>
  </tr>
  <tr>
    <td>FrequencyGoverner</td>
    <td>Performance</td>
  </tr>
</tbody>
</table>
