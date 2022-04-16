# NLP Executor：A baremetal inference engine for Natural Language Processing(NLP) Models
NLP Executor is an inference executor for Natural Language Processing (NLP) models, providing the optimal performance by quantization and sparsity. The executor is a baremetal reference engine of NLP Toolkit and supports typical NLP models.

## Deployment Architecture
The executor supports model optimization and high performance kernel for CPU.
<a target="_blank" href="docs/imgs/engine_infrastructure.png">
  <img src="nlp_toolkit/backends/nlp_executor/docs/imgs/infrastructure.png" alt="Infrastructure" width=762 height=672>
</a>  

## Installation
Just support Linux operating system for now.

### 0. Prepare environment and requirement
```
# prepare your env
conda create -n <env name> python=3.7
conda install cmake --yes
conda install absl-py --yes
```

### 1. install neural-compressor

NLP executor depends on neural-compressor, but we will decouple them in the future.

```
pip install neural-compressor
```

### 2. install C++ binary by deploy executor

```
cd <project folder/nlp_toolkit/>
python setup.py install/develop
```
Then in the nlp_toolkit/build folder, you will get the `nlp_executor`, `engine_py.cpython-37m-x86_64-linux-gnu.so` and `libexecutor.so`. 
The first one is used for pure c++ APIs, and the second is used for Python APIs, they all need the `libexecutor.so`.

## Generate the bert model intermediate representations, that are yaml and bin files

```
from nlp_executor.compile import compile
model = compile('/path/to/your/model')
model.save('/ir/path')
```
Now the executor support tensorflow and onnx model conversion.

## Use case

### 1. C++ API

`./nlp_executor --config=<the generated yaml file path> --weight=<the generated bin file path> --batch_size=32 --iterations=20`

You can use the `numactl` command to bind cpu cores and open multi-instances:

`OMP_NUM_THREADS=4 numactl -C '0-3' ./nlp_executor ...`

Open/Close Log:(GLOG_minloglevel=1/GLOG_minloglevel=2)

 `export GLOG_minloglevel=2 ./nlp_executor ...`


### 2. Python APIs

If you use pip install -e . to install the executor in your current folder, please make sure to export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/libexecutor.so

You can skip the step if install with other ways .

```python
from engine_py import Model
# load the model, config_path:path of generated yaml, weight_path: path of generated bin
model = Model(config_path, weight_path)
# use model.forward to do inference
out = model.forward([input_ids, segment_ids, input_mask])
```

The `input_ids`, `segment_ids` and `input_mask` are the input numpy array data of a bert model, which have size (batch_size, seq_len). 
Note that the `out` is a list contains the bert model output numpy data (`out=[output numpy data]`). 
