# Assisted Generation

Intel Extension for Transformers supports assisted generation (aka speculative decoding) using the Hugging Face API, to speedup inference by up to 3x.

Assisted generation uses an assistant model with the same tokenizer (ideally a much smaller model) to greedily generate a few candidate tokens. The main model then validates the candidate tokens in a single forward pass, which speeds up the decoding process.  Note that the speedup increases as the size of the main model increases.

See [here](https://huggingface.co/blog/assisted-generation) for more info on assisted generation.


## Setup

Create a conda environment (recommended):

```bash
conda create -n llm python=3.9 -y && conda activate llm
```

Copy these commands into the shell:

```bash
conda install ninja mkl mkl-include -y
conda install gperftools -c conda-forge -y

python -m pip install torch==2.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel_extension_for_pytorch

# install others deps
python -m pip install cpuid transformers accelerate onnx sentencepiece
python -m pip install --no-deps optimum
python -m pip install --no-deps git+https://github.com/huggingface/optimum-intel.git@main

# Setup Environment Variables for best performance on Xeon
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
```

## Usage 

```bash
# Support single socket and multiple sockets
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_assisted_generation.py -m facebook/opt-13b
```

These are the default assistant models used in the script:

```python
"OPTForCausalLM": "facebook/opt-125m",
"LlamaForCausalLM": "PY007/TinyLlama-1.1B-intermediate-step-715k-1.5T",
"GPTBigCodeForCausalLM": "bigcode/tiny_starcoder_py"
```

To specify a different assistant model:

```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_assisted_generation.py -m "facebook/opt-13b" --assistant-model "facebook/opt-350m"
```



To turn off assisted generation feature (for benchmarking purposes):

```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_assisted_generation.py -m "facebook/opt-13b" --no-assisted
```
