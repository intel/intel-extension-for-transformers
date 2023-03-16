Step-by-Step
=========
This document describes the end-to-end workflow for Text-to-image generative AI models [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) and [runwayml/stable-diffusion-v1-5](https://github.com/runwayml/stable-diffusion) across the Neural Engine backend.

# Prerequisite

## 1. Installation
### 1.1 Install python environment
Create a new python environment
```shell
conda create -n <env name> python=3.8
conda activate <env name>
```
Make sure you have the autoconf installed. 
Also, `gcc` higher than 9.0, `cmake` higher than 3 is required.
```shell
gcc -v
cmake --version
conda install cmake
sudo apt install autoconf
```
Install Intel® Extension for Transformers, please refer to [installation](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/installation.md)
```shell
# Install from pypi
pip install intel-extension-for-transformers

# Install from source code
cd <intel_extension_for_transformers_folder>
git submodule update --init --recursive
python setup.py install
```
Install required dependencies for examples
```shell
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/sst2/bert_mini
pip install -r requirements.txt
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11


### 1.2 Environment variables preload libjemalloc.so can improve the performance when multi instances.
```
export LD_PRELOAD=<intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```
Using weight sharing can save memory and improve the performance when multi instances.
```
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
## 2. Get and Export Pretrained Model

The pretrained model [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) and [runwayml/stable-diffusion-v1-5](https://github.com/runwayml/stable-diffusion) provied by diffusers might be the same in the default config, so the former is chosen here as an example.

### 2.1 Get models

Get a fp32 ONNX model from the hugginface diffusers module, command as follows:

```python
python prepare_model.py --input_model=CompVis/stable-diffusion-v1-4 --output_path=./model
```

By setting --bf16=True to get fp32 and bf16 models together.
```python
python prepare_model.py --input_model=CompVis/stable-diffusion-v1-4 --output_path=./model --bf16=True
```

### 2.2 Compile Models
Convert three onnx sub-models of the stable diffusion to Nerual Engine IRs.

```python
# 1. text encoder
python export_ir.py --onnx_model=./model/text_encoder/model.onnx --pattern_config=text_encoder_pattern.conf --output_path=./ir/text_encoder/

# 2. unet
python export_ir.py --onnx_model=./model/unet/model.onnx --pattern_config=unet_pattern.conf --output_path=./ir/unet/

# 3. vae_decoder
python export_ir.py --onnx_model=./model/vae_decoder/model.onnx --pattern_config=vae_decoder_pattern.conf --output_path=./ir/vae_decoder/
```
Note:
> 1. using "export LOGLEVEL=DEBUG" to check all matched pattern nodes.

## 2.3 Run Stable Diffusion

Text-to-image: using one sentence to create a picture.

```python
python run_executor.py --ir_path=./ir
```

Note: 
> 1. The default pretrained model is "CompVis/stable-diffusion-v1-4".
> 2. The default prompt is "a photo of an astronaut riding a horse on mars" and the default output name is "astronaut_rides_horse.png".
> 3. The ir directory should include three IRs for text_encoder, unet and vae_decoder.

## Benchmark

### 2.1 Performance
Python API command as follows:
  ```shell
  GLOG_minloglevel=2 python run_executor.py --mode=performance
  ```
