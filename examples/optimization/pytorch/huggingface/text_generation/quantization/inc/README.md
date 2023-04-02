Step-by-Step​
============
The scripts `run_text.py` provide two quantization approaches respectively (PostTrainingDynamic and PostTrainingStatic) based on [Intel® Extension for Transformers](https://github.com/intel/intel-extension-for-transformers).

# Prerequisite​
## 1. Create Environment​
Recommend python 3.7 or higher version is recommended. The dependent packages are listed in requirements, please install them as follows,

```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Run
## 1. Quantization
**Text Generation**

``` bash
python run_text.py \
    --model_name_or_path bigscience/bloom-560m \
    --dataset_name lambada \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --overwrite_output_dir

```

## 2. Validated Model List

<table>
<thead>
  <tr>
    <th>Topology Name</th>
    <th>Model Name</th>
    <th>Dataset/Task Name</th>
    <th>Quantization Approach</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td>bloom_text_static</td>
    <td><a href="https://huggingface.co/bigscience/bloom-560m">bigscience/bloom-560m</a></td>
    <td><a href="https://huggingface.co/datasets/lambada">lambada</td>
    <td>PostTrainingStatic</td>
  </tr>
    <tr>
    <td>bloom_text_dynamic</td>
    <td><a href="https://huggingface.co/bigscience/bloom-560m">bigscience/bloom-560m</a></td>
    <td><a href="https://huggingface.co/datasets/lambada">lambada</td>
    <td>PostTrainingDynamic</td>
  </tr>
</tbody>
</table>

## 3. Bash Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```
