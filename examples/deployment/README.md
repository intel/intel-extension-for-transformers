# Deployment
NLP Toolkit provides multiple reference deployments: 1) [**Neural Engine**](neural_engine); 2) [IPEX (Coming Soon)](ipex/).

## Neural Engine
Neural Engine can provide the optimal performance of extremely compressed NLP models, the optimization is both from HW and SW.It's a reference deployment for NLPToolkit, we will enable other backends.

Supported Examples
| Question-Answering | Text-Classification |
|:---:|:---:|
|[Bert-large (SQUAD)](https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/tree/develop/examples/deployment/neural_engine/squad/bert_large)|[Bert-mini (SST2)](https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/tree/develop/examples/deployment/neural_engine/sst2/bert_mini)</br> [MiniLM (SST2)](https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/tree/develop/examples/deployment/neural_engine/sst2/minilm_l6_h384_uncased)</br> [Distilbert (SST2)](https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/tree/develop/examples/deployment/neural_engine/sst2/distilbert_base_uncased) </br> [Distilbert (Emotion)](https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/tree/develop/examples/deployment/neural_engine/emotion/distilbert_base_uncased) </br> [Bert-base (MRPC)](https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/tree/develop/examples/deployment/neural_engine/mrpc/bert_base)</br> [Bert-mini (MRPC)](https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/tree/develop/examples/deployment/neural_engine/mrpc/bert_mini)</br>[Distilbert (MRPC)](https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/tree/develop/examples/deployment/neural_engine/mrpc/distilbert_base_uncased)</br> [Roberta-base (MRPC)](https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/tree/develop/examples/deployment/neural_engine/mrpc/roberta_base)</br>|

### Architecture
Here is the architecture of reference deployment:
<a target="_blank" href="../../nlp_toolkit/backends/nlp_executor/docs/imgs/infrastructure.png">
  <img src="../../nlp_toolkit/backends/neural_engine/docs/imgs/infrastructure.png" alt="Infrastructure" width=762 height=672>
</a>  

#### Installation
Linux is supported only.

##### 0. Prepare environment and requirement
```
# prepare your env
conda create -n <env name> python=3.7
conda install cmake --yes
conda install absl-py --yes
```

##### 1. Install neural-compressor

Install Intel Neural Compressor as a pre-condition

```
pip install neural-compressor
```

##### 2. Build neural engine

```
cd <project folder/nlp_toolkit/>
python setup.py install/develop
```
After succesful build, you will see `neural_engine` in the nlp_toolkit/build folder. 

##### 3. Generate optimal BERT model

```
from neural_engine.compile import compile
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

If you use pip install -e . to install the neural engine in your current folder, please make sure to export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/libneural_engine.so.

```python
from engine_py import Model
# load the model, config_path:path of generated yaml, weight_path: path of generated bin
model = Model(config_path, weight_path)
# use model.forward to do inference
out = model.forward([input_ids, segment_ids, input_mask])
```

The `input_ids`, `segment_ids` and `input_mask` are the input numpy array data of BERT model, and the input dimension is (batch_size x seq_len). 
Note that the `out` is a list contains the bert model output numpy data (`out=[output numpy data]`). 

## IPEX
Intel® Extension for PyTorch* extends PyTorch with optimizations for extra performance boost on Intel hardware. Sample deployment is coming soon.

