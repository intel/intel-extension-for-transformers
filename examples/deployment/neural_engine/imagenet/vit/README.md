# Step-by-Step

# Prerequisite

### 1. Installation
1.1 Install python environment
Create a new python environment
```shell
conda create -n <env name> python=3.8
conda activate <env name>
```
Check the gcc version using $gcc-v, make sure the gcc version is higher than 9.0.
If not, you need to update gcc by yourself.
Make sure you have the autoconf installed.
If not, you need to install autoconf by yourself.
Make sure the cmake version is 3 rather than 2.
If not, you need to install cmake.
```shell
cmake --version
conda install cmake
sudo apt install autoconf
```
Install NLPTookit from source code
```shell
cd <NLP_Toolkit_folder>
git submodule update --init --recursive
python setup.py install
```

### 2. Run the code to conduct model quantization and model conversion to ONNX format

### 2.1 Automatically download the pretrained model and conduct the operation
```shell
python -u model_quant_convert.py --model_name_or_path google/vit-large-patch16-224 --per_device_eval_batch_size 8 --remove_unused_columns False --output_dir ./output/ --use_auth_token --do_eval --no_cuda --overwrite_output_dir --accuracy_only  --tune --do_train --dataset_name imagenet-1k
```

### 2.2 Save the cache data
Considering the size of the dataset is too large, we recommend the user to save the cache data for the ease of next running:
```shell
dataset.save_to_disk("./cached-2k-imagenet-1k-datasets")
```
And the dataset could be reloaded by:
```shell
dataset = datasets.load_from_disk("./cached-2k-imagenet-1k-datasets")
```

### 3 Check the performance of the converted ONNX model
The converted onnx model could be found in --output_dir. The user could check the accuracy with:
```shell
python model_eval.py --model_name_or_path google/vit-large-patch16-224 --per_device_eval_batch_size 8 --remove_unused_columns False --output_dir ./output/ --overwrite_output_dir --dataset_name imagenet-1k --mode onnx
```