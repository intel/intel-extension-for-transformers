
# Build RAG (retriveval augment generation) example with Intel® Extension for Transformers neural-chat on Intel GPU

# 1. Setup Environment

## prerequisite
GPU driver and oneAPI Base Toolkit 2024.0 is required.

## 1.1 Install intel-extension-for-transformers

```
conda create -n itrex-rag python=3.9
conda activate itrex-rag
```

## 1.2 Install neural-chat and retrieval dependency

```
git clone https://github.com/intel/intel-extension-for-transformers.git ~/itrex
cd ~/itrex/intel_extension_for_transformers/neural_chat/examples/quick_start/rag
sh install_rag_gpu.sh
```

# 2. Run the RAG in command mode

## Usage

Example 1. Run the example with RAG

Set oneAPI environment. Make sure oneAPI Base Toolkit version is 2024.0.

```
source /opt/intel/oneapi/setvars.sh

cd ~/itrex/intel_extension_for_transformers/neural_chat/examples/quick_start/rag
python retrieval.py
```

Example 2. Run the example disable RAG

```
python retrieval.py --no-retrieval
```
