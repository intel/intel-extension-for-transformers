# Deployment
Intel Extension for Transformers provides multiple reference deployments: 1) [**Neural Engine**](neural_engine); 2) [IPEX](ipex/).

## Neural Engine
Neural Engine can provide the optimal performance of extremely compressed transformer based models, the optimization is both from HW and SW. It's a reference deployment for Intel Extension for Transformers, we will enable other backends.

Supported Examples
| Question-Answering | Text-Classification |
|:---:|:---:|
|[Bert-large (SQUAD)](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/deployment/neural_engine/squad/bert_large)|[Bert-mini (SST2)](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/deployment/neural_engine/sst2/bert_mini)</br> [MiniLM (SST2)](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/deployment/neural_engine/sst2/minilm_l6_h384_uncased)</br> [Distilbert (SST2)](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/deployment/neural_engine/sst2/distilbert_base_uncased) </br> [Distilbert (Emotion)](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/deployment/neural_engine/emotion/distilbert_base_uncased) </br> [Bert-base (MRPC)](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/deployment/neural_engine/mrpc/bert_base)</br> [Bert-mini (MRPC)](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/deployment/neural_engine/mrpc/bert_mini)</br>[Distilbert (MRPC)](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/deployment/neural_engine/mrpc/distilbert_base_uncased)</br> [Roberta-base (MRPC)](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/deployment/neural_engine/mrpc/roberta_base)</br>|

#### Installation
Windows and Linux are supported.

##### 0. Prepare environment and requirement
```
# prepare your env
conda create -n <env name> python=3.7
conda install cmake --yes
conda install absl-py --yes
```

##### 1. Install neural-compressor

Install Intel Neural Compressor as a pre-condition, recommend version is 1.14.2

```
pip install neural-compressor==1.14.2
```

##### 2. Build neural engine

```
cd <project folder/intel_extension_for_transformers/>
python setup.py install/develop
```

##### 3. Generate optimal BERT model

```
from intel_extension_for_transformers.backends.neural_engine.compile import compile
model = compile('/path/to/your/model')
model.save('/ir/path')
```
Note that Neural Engine supports TensorFlow and ONNX models.

##### 4. Deployment

###### 4.1. C++ API

`./neural_engine --config=<path to yaml file> --weight=<path to bin file> --batch_size=32 --iterations=20`

You can use the `numactl` command to bind cpu cores and open multi-instances:

`OMP_NUM_THREADS=4 numactl -C '0-3' ./neural_engine ...`

Open/Close Log:(GLOG_minloglevel=1/GLOG_minloglevel=2)

`export GLOG_minloglevel=2 ./neural_engine ...`


###### 4.2. Python API

If you use python setup.py install to install the neural engine in your current folder, then you can use python api as following.

```python
from intel_extension_for_transformers.backends.neural_engine.compile import compile
# load the model
graph = compile('./model_and_tokenizer/int8-model.onnx')
# use graph.inference to do inference
out = graph.inference([input_ids, segment_ids, input_mask])
# dump the neural engine IR to file
graph.save('./ir')
```

The `input_ids`, `segment_ids` and `input_mask` are the input numpy array data of model, and the input dimension is variable. 
Note that the `out` is a dict contains the bert model output tensor name and numpy data (`out={output name : numpy data}`). 

##### 5. Analyze operator performance

If you want to analyze performance of each operator, just export ENGINE_PROFILING=1 and export INST_NUM=<inst_num>.
It will dump latency of each operator to <curr_path>/engine_profiling/profiling_<time>_<inst_count>.csv and each iteration to <curr_path>/engine_profiling/profiling_<time>_<inst_count>.json.

## IPEX
Intel® Extension for PyTorch* extends PyTorch with optimizations for extra performance boost on Intel hardware.

