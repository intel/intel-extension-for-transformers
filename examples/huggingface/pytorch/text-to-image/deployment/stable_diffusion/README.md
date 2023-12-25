Step-by-Step
=========
This document describes the end-to-end workflow for Text-to-image generative AI models across the Neural Engine backend.

Supported Text-to-image Generative AI models:
1. [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
2. [runwayml/stable-diffusion-v1-5](https://github.com/runwayml/stable-diffusion)
3. [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
4. [instruction-tuning-sd](https://huggingface.co/instruction-tuning-sd)
    * [scratch-low-level-img-proc](https://huggingface.co/instruction-tuning-sd/scratch-low-level-img-proc)
    * [scratch-cartoonizer](https://huggingface.co/instruction-tuning-sd/scratch-cartoonizer)
    * [cartoonizer](https://huggingface.co/instruction-tuning-sd/cartoonizer)
    * [low-level-img-proc](https://huggingface.co/instruction-tuning-sd/low-level-img-proc)

The inference and accuracy of the above pretrained models are verified in the default configs.

# Prerequisite

## Prepare Python Environment
Create a python environment, optionally with autoconf for jemalloc support.
```shell
conda create -n <env name> python=3.8 [autoconf]
conda activate <env name>
```
>**Note**: Make sure pip <=23.2.2

Check that `gcc` version is higher than 9.0.
```shell
gcc -v
```

Install Intel® Extension for Transformers, please refer to [installation](/docs/installation.md).
```shell
# Install from pypi
pip install intel-extension-for-transformers

# Or, install from source code
cd <intel_extension_for_transformers_folder>
pip install -r requirements.txt
pip install -v .
```

Install required dependencies for this example
```shell
cd <intel_extension_for_transformers_folder>/examples/huggingface/pytorch/text-to-image/deployment/stable_diffusion

pip install -r requirements.txt
pip install transformers==4.34.1
pip install diffusers==0.12.1
```
>**Note**: Please use transformers no higher than 4.34.1

## Environment Variables (Optional)
```shell
# Preload libjemalloc.so may improve the performance when inference under multi instance.
conda install jemalloc==5.2.1 -c conda-forge -y
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so

# Using weight sharing can save memory and may improve the performance when multi instances.
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
>**Note**: This step is optional.
# End-to-End Workflow
## 1. Prepare Models

The stable diffusion mainly includes three sub models: 
1. Text Encoder 
2. Unet 
3. Vae Decoder.

Here we take the [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) as an example.

### 1.1 Download Models
Export FP32 ONNX models from the hugginface diffusers module, command as follows:

```python
python prepare_model.py --input_model=CompVis/stable-diffusion-v1-4 --output_path=./model
```

By setting --bf16 to export FP32 and BF16 models.
```python
python prepare_model.py --input_model=CompVis/stable-diffusion-v1-4 --output_path=./model --bf16
```

For INT8 quantized mode, we **only support runwayml/stable-diffusion-v1-5** for now.
You need to get a quantized INT8 model first through QAT, Please refer the [link](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/text-to-image/quantization/qat/README.md).
Then by setting --qat_int8 to export INT8 models, to export INT8 model.
```python
python prepare_model.py --input_model=runwayml/stable-diffusion-v1-5 --output_path=./model --qat_int8
```

### 1.2 Compile Models

Export three FP32 onnx sub models of the stable diffusion to Nerual Engine IR.

```bash
# running the follow bash comand to get all IR.
bash export_model.sh --input_model=model --precision=fp32
```

Export three BF16 onnx sub models of the stable diffusion to Nerual Engine IR.

```bash
# running the follow bash comand to get all IR.
bash export_model.sh --input_model=model --precision=bf16
```

Export mixed FP32 & dynamic quantized Int8 IR.

```bash
bash export_model.sh --input_model=model --precision=fp32 --cast_type=dynamic_int8
```

Export mixed BF16 & QAT quantized Int8 IR.
```bash
bash export_model.sh --input_model=model --precision=qat_int8
```

## 2. Performance

Python API command as follows:
```python
# FP32 IR
python run_executor.py --ir_path=./fp32_ir --mode=latency --input_model=CompVis/stable-diffusion-v1-4

# Mixed FP32 & dynamic quantized Int8 IR.
python run_executor.py --ir_path=./fp32_dynamic_int8_ir --mode=latency --input_model=CompVis/stable-diffusion-v1-4

# BF16 IR
python run_executor.py --ir_path=./bf16_ir --mode=latency --input_model=CompVis/stable-diffusion-v1-4

# QAT INT8 IR
python run_executor.py --ir_path=./qat_int8_ir --mode=latency --input_model=runwayml/stable-diffusion-v1-5
```

## 3. Accuracy
Frechet Inception Distance(FID) metric is used to evaluate the accuracy. This case we check the FID scores between the pytorch image and engine image.

By setting --accuracy to check FID socre.
Python API command as follows:
```python
# FP32 IR
python run_executor.py --ir_path=./fp32_ir --mode=accuracy --input_model=CompVis/stable-diffusion-v1-4

# Mixed FP32 & dynamic quantized Int8 IR
python run_executor.py --ir_path=./fp32_dynamic_int8_ir --mode=accuracy --input_model=CompVis/stable-diffusion-v1-4

# BF16 IR
python run_executor.py --ir_path=./bf16_ir --mode=accuracy --input_model=CompVis/stable-diffusion-v1-4

# QAT INT8 IR
python run_executor.py --ir_path=./qat_int8_ir --mode=accuracy --input_model=runwayml/stable-diffusion-v1-5
```

## 4. Try Text to Image

### 4.1 Text2Img

Try using one sentence to create a picture!

```python
# Running FP32 models or BF16 models, just import differnt IR.
# FP32 models
# Note: 
# 1. Using --image to set the path of your image, here we use the default download link.
# 2. The default image is "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png".
# 3. The default prompt is "Cartoonize the following image".

python run_executor.py --ir_path=./fp32_ir --input_model=CompVis/stable-diffusion-v1-4
```
![picture1](./images/astronaut_rides_horse.png)

```python
# BF16 models
python run_executor.py --ir_path=./bf16_ir --input_model=CompVis/stable-diffusion-v1-4
```
![picture2](./images/astronaut_rides_horse_from_engine_1.png)


### 4.2 Img2Img: instruction-tuning-sd

Try using one image and prompts to create a new picture!

```python
# Running FP32 models or BF16 models, just import differnt IR.
# BF16 models
python run_executor.py --ir_path=./bf16_ir --input_model=instruction-tuning-sd/cartoonizer --pipeline=instruction-tuning-sd --prompts="Cartoonize the following image" --steps=100
```

Original image:

![picture3](./images/mountain.png)

Cartoonized image:

![picture4](./images/mountain_cartoonized.png)


## 5. Validated Result


### 5.1 Latency (s)


Input: a photo of an astronaut riding a horse on mars

Batch Size: 1


| Model | FP32 | BF16 | 
|---------------------|:----------------------:|-----------------------|
| [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | 10.33 (s) | 3.02 (s) |

> Note: Performance results test on ​​06/09/2023 with Intel(R) Xeon(R) Platinum 8480+.
Performance varies by use, configuration and other factors. See platform configuration for configuration details. For more complete information about performance and benchmark results, visit www.intel.com/benchmarks



### 5.2 Platform Configuration


<table>
<tbody>
  <tr>
    <td>Manufacturer</td>
    <td>Quanta Cloud Technology Inc</td>
  </tr>
  <tr>
    <td>Product Name</td>
    <td>QuantaGrid D54Q-2U</td>
  </tr>
  <tr>
    <td>OS</td>
    <td>CentOS Stream 8</td>
  </tr>
  <tr>
    <td>Kernel</td>
    <td>5.16.0-rc1-intel-next-00543-g5867b0a2a125</td>
  </tr>
  <tr>
    <td>Microcode</td>
    <td>0x2b000111</td>
  </tr>
  <tr>
    <td>IRQ Balance</td>
    <td>Eabled</td>
  </tr>
  <tr>
    <td>CPU Model</td>
    <td>Intel(R) Xeon(R) Platinum 8480+</td>
  </tr>
  <tr>
    <td>Base Frequency</td>
    <td>2.0GHz</td>
  </tr>
  <tr>
    <td>Maximum Frequency</td>
    <td>3.8GHz</td>
  </tr>
  <tr>
    <td>CPU(s)</td>
    <td>224</td>
  </tr>
  <tr>
    <td>Thread(s) per Core</td>
    <td>2</td>
  </tr>
  <tr>
    <td>Core(s) per Socket</td>
    <td>56</td>
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
