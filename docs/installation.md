# Installation

1. [System Requirement](#system-requirements)

2. [Install from Pypi](#install-from-pypi)

3. [Install from Source](#install-from-source)

    3.1. [Prerequisites](#prerequisites)

    3.2. [Install Intel Extension for Transformers](#install-intel-extension-for-transformers)

4. [Validated Environment](#validated-environment)

    4.1. [Validated Hardware Environment](#validated-hardware-environment)

    4.2. [Validated Software Environment](#validated-software-environment)


## System Requirements
For NeuralChat usage, please make sure if you have below system libraries installed:

### Ubuntu 20.04/22.04
```bash
apt-get update
apt-get install -y ffmpeg
apt-get install -y libgl1-mesa-glx libgl1-mesa-dev
apt-get install -y libsm6 libxext6
```    

### Centos 8
```bash
yum update -y
yum install -y mesa-libGL mesa-libGL-devel
yum install -y libXext libSM libXrender
```

For ffmpeg, please refer to [how-to-install-ffmpeg-on-centos-rhel-8](https://computingforgeeks.com/how-to-install-ffmpeg-on-centos-rhel-8/)     


## Install from Pypi
Binary builds for python 3.9, 3.10 and 3.11 are available in Pypi

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
- Python version: 3.9 or 3.10 or 3.11
- GCC >= version 10 (on Linux)
- Visual Studio (on Windows)

 >**Note**: If your system only have python3 or you meet error `python: command not found`, please run `ln -sf $(which python3) /usr/bin/python`.
 

### Install Intel Extension for Transformers
```Bash
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
pip install -r requirements.txt
# Install intel_extension_for_transformers
pip install -v .
```

## Validated Environment

### Validated Hardware Environment
Intel® Extension for Transformers supports the following HWs:

* Intel Xeon Scalable processor (Sapphire Rapids, Icelake, ...etc)
* Intel Gaudi2
* Intel Core Processors
* Intel Xeon CPU Max Series

### Validated Software Environment

* OS version: CentOS 8.4, Ubuntu 20.04
* Python version: 3.9, 3.10, 3.11  

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
* Python version: 3.9, 3.10, 3.11 

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
    <td><a href=https://download.pytorch.org/whl/torch_stable.html>2.1.0+cpu</a><br>
  </tr>
</tbody>
</table>
