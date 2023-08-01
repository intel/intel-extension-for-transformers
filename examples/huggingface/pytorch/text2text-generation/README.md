<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
Step-by-Step
============
This document describes the step-by-step instructions to run seq2seq generation on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with PyTorch and Intel® Extension for PyTorch.

The scripts `run_seq2seq_generation.py` provide the quantization method based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).
In this example, we support T5 series models, like: T5-base, T5-small, T5-large and flan-T5...... .

# Prerequisite​
## 1. Create Environment​
Recommend python 3.7 or higher version is recommended. The dependent packages are listed in requirements, please install them as follows,

Here is how to install intel-extension-for-pytorch from source, It will also automatically install nightly version torch, torchvision, ad torchaudio.
```shell
wget https://raw.githubusercontent.com/intel/intel-extension-for-pytorch/scripts/compile_bundle.sh
bash compile_bundle.sh
pip install requirements.txt
```

# Run
## T5-base tag generation

```bash
python run_seq2seq_generation.py \
    --model_type=t5 \
    --model_name_or_path=fabiochiu/t5-base-tag-generation \
    --dataset=NeelNanda/pile-10k \
    --quantize \
    --ipex # Use intel-extension-for-pytorch backend \
    --sq # Use smoothquant recipe for quantization
    --alpha 0.7 # smoothquant args.
```

## Flan-T5-large text generation
```bash
python run_seq2seq_generation.py \
    --model_type=t5 \
    --model_name_or_path=google/flan-t5-large \
    --dataset=NeelNanda/pile-10k \
    --quantize \
    --ipex # Use intel-extension-for-pytorch backend \
    --sq # Use smoothquant recipe for quantization
    --alpha 0.7 # smoothquant args.
```
