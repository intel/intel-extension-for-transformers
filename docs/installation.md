# Installation

1. [Install from Pypi](#install-from-pypi)

2. [Install from Source](#install-from-source)

    2.1. [Prerequisites](#prerequisites)

    2.2. [Install Intel Extension for Transformers](#install-intel-extension-for-transformers)

3. [System Requirements](#system-requirements)

    3.1. [Validated Hardware Environment](#validated-hardware-environment)

    3.2. [Validated Software Environment](#validated-software-environment)

## Install from Pypi
Binary builds for python 3.8, 3.9 and 3.10 are available in Pypi

>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11

```Bash
# install stable basic version from pypi
pip install intel-extension-for-transformers
```

```Bash
# install stable basic version from conda
conda install -c intel intel_extension_for_transformers
```

## Install from Source

### Prerequisites
The following prerequisites and requirements must be satisfied for a successful installation:
- Python version: 3.8 or 3.9 or 3.10
- GCC >= version 8 (on Linux)
  - version 11, if use the bf16-related features of the itrex backend
  - version 13, if use the fp16-related features of the itrex backend
- Visual Studio (on Windows)

### Install Intel Extension for Transformers
```Bash
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
pip install -r requirements.txt
# Install intel_extension_for_transformers
pip install -v .
```

## System Requirements
### Validated Hardware Environment
Intel® Extension for Transformers supports systems based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64) that are specifically optimized for the following CPUs:

* Intel Xeon Scalable processor (formerly Cascade Lake, Icelake)
* Future Intel Xeon Scalable processor (code name Sapphire Rapids)

### Validated Software Environment

* OS version: CentOS 8.4, Ubuntu 20.04
* Python version: 3.8, 3.9, 3.10  

<table class="docutils">
<thead>
  <tr>
    <th>Framework</th>
    <th>Intel TensorFlow</th>
    <th>PyTorch</th>
    <th>IPEX</th>
  </tr>
</thead>
<tbody>
  <tr align="center">
    <th>Version</th>
    <td class="tg-7zrl"><a href=https://github.com/Intel-tensorflow/tensorflow/tree/v2.13.0>2.13.0</a><br>
    <a href=https://github.com/Intel-tensorflow/tensorflow/tree/v2.12.0>2.12.0</a><br>
    <td class="tg-7zrl"><a href=https://download.pytorch.org/whl/torch_stable.html>2.1.0+cpu</a><br>
    <a href=https://download.pytorch.org/whl/torch_stable.html>2.0.0+cpu</a><br>
    <td class="tg-7zrl"><a href=https://github.com/intel/intel-extension-for-pytorch/tree/v2.1.0+cpu>2.1.0+cpu</a><br>
    <a href=https://github.com/intel/intel-extension-for-pytorch/tree/v2.0.0+cpu>2.0.0+cpu</a></td>
  </tr>
</tbody>
</table>

* OS version: Windows 10
* Python version: 3.8, 3.9, 3.10  

<table class="docutils">
<thead>
  <tr>
    <th>Framework</th>
    <th>Intel TensorFlow</th>
    <th>PyTorch</th>
  </tr>
</thead>
<tbody>
  <tr align="center">
    <th>Version</th>
    <td><a href=https://github.com/Intel-tensorflow/tensorflow/tree/v2.13.0>2.13.0</a><br>
    <td><a href=https://download.pytorch.org/whl/torch_stable.html>2.0.0+cpu</a><br>
  </tr>
</tbody>
</table>
