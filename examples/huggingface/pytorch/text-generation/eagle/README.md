# Eagle - Speculative Sampling

Intel Extension for Transformers supports the EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) which is a speculative sampling method that improves text generation speed.

See [here](https://arxiv.org/abs/2401.15077) to view the paper and [here](https://github.com/SafeAILab/EAGLE) for more info on EAGLE code.


## Setup and installation 

With pip: (recommended)

```bash
pip install eagle-llm
```

From the source:

```bash
git clone https://github.com/SafeAILab/EAGLE.git
cd EAGLE
pip install -e .
```


## Usage 

The script accepts several command-line arguments:

- -d or --device: Target device for text generation (default: "cpu", options: "cpu", "xpu", "cuda").
- -t or --dtype: Data type for text generation (default: "float32", options: "float32", "float16", "bfloat16").
- --max_new_tokens: Number of max new tokens for text generation (default: 512).
- --use_eagle: Use EAGLE model for generation (default: False).

```bash
python3 eagle_example.py
python eagle_example.py -d xpu --max_new_tokens 1024 --use_eagle

```

The default base model is set to "meta-llama/Llama-2-7b-chat-hf", you can change it in the script by reassigning the variable "base_model_path" to model of your choice


# Features
- Dynamic Device and Data Type Configuration: The script allows specifying the target device and data type for text generation, supporting CPU, XPU, and CUDA devices, as well as different data types.
- EAGLE Model Integration: The script demonstrates how to integrate the EAGLE model for enhanced text generation capabilities.
- Performance Measurement: The script measures the performance of the text generation process, including the tokens per second (TPS) based on the total new tokens and total time.


# Results

We conducted benchmarking tests on both CPU and XPU environments.

On GPU, llama2-7b-chat + EAGLE outperforms llama2-7b-chat by 3x speed in generating new tokens.
On CPU, llama2-7b-chat + EAGLE outperforms llama2-7b-chat by 1.75x speed in generating new tokens.

This highlights the superior capabilities of the EAGLE in handling the text generation task.

