Deploy and Integration
=====
In this tutorial, we will deploy a TF/ONNX model using Engine inference or through Manual customized yaml and weight binary to use Engine inference.

[1. Architecture](#1-architecture)

[2. Deploy a TF/ONNX model using Engine inference](#2-deploy-a-tfonnx-model-using-engine-inference)

[3. Manual customized yaml and weight binary to use Engine inference](#3-manual-customized-yaml-and-weight-binary-to-use-engine-inference)

[4. Integrate Neural Engine as Backend](#4-Integrate-Neural-Engine-as-Backend)

## 1. Architecture
Neural Engine support model optimizer, model executor and high performance kernel for multi device.

<a target="_blank" href="imgs/infrastructure.png">
  <img src="imgs/infrastructure.png" alt="Architecture" width=762 height=672>
</a>

## 2. Deploy a TF/ONNX model using Engine inference

### Generate the Engine Graph through TF/ONNX model

Only support TensorFlow and ONNX models for now.

```python
from intel_extension_for_transformers.backends.neural_engine.compile import compile
model = compile('/path/to/your/model')
model.save('/ir/path')   # Engine graph could be saved to path
```

Engine graph could be saved as yaml and weight bin.

### Run the inference by Engine

```python
model.inference([input_ids, segment_ids, input_mask])  # input should be numpy array data
```

The `input_ids`, `segment_ids` and `input_mask` are the input numpy array data of a bert model, which have size (batch_size, seq_len). Note that the `out` is a dict contains the output tensor name and value(numpy array).

## 3. Manual customized yaml and weight binary to use Engine inference

### Build the yaml and weight binary

Engine could parse yaml structure network and load the weight from binary to do inference, yaml should like below

```yaml
model:
  name: bert_model
  operator:
    input_data:
      type: Input                # define the input and weight shape/dtype/location
      output:
        input_ids:0:
          dtype: int32
          shape: [-1, -1]
        segment_ids:0:
          dtype: int32
          shape: [-1, -1]
        input_mask:0:
          dtype: int32
          shape: [-1, -1]
        bert/embeddings/word_embeddings:0:
          dtype: fp32
          shape: [30522, 1024]
          location: [0, 125018112]
          ....
    padding_sequence:                   # define the operators type/input/output/attr
      type: PaddingSequence
      input:
        input_mask:0: {}
      output:
        padding_sequence:0: {}
      attr:
        dst_shape: -1,16,0,-1
        dims: 1
    bert/embeddings/Reshape:
      type: Reshape
      input:
        input_ids:0: {}
      output:
        bert/embeddings/Reshape:0: {}
      attr:
        dst_shape: -1
    ....
    output_data:                       # define the output tensor
      type: Output
      input:
        logits:0: {}
```
All input tensors are in an operator typed Input. But slightly difference is some tensors have location while others not. A tensor with location means that is a frozen tensor or weight, it's read from the bin file. A tensor without location means it's activation, that should feed to the model during inference.

### Run the inference by Engine

Parse the yaml and weight bin to Engine Graph throught Python API

```python
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
model = Graph()
model.graph_init('./ir/conf.yaml', './ir/model.bin')
input_data = [input_0, input_1, input_2]
out = model.inference(input_data)
```

You can also use C++ API
```shell
neural_engine --config=<path to yaml file> --weight=<path to bin file> --batch_size=32 --iterations=20
```
Using the `numactl` command to bind cpu cores and open multi-instances:
```shell
OMP_NUM_THREADS=4 numactl -C '0-3' neural_engine ...
```
Same as the previous session, the ***input_data*** should be numpy array data as a list, and ***out*** is a dict which pair the output tensor name and value(numpy array).

>**Note**: Extensive log information is available if build with Debug. We use [glog](https://github.com/google/glog) for logging and respect [its environment variables](https://github.com/google/glog#setting-flags) such as `GLOG_minloglevel`.

## 4. Integrate Neural Engine as Backend
Nerual Engine can also be integrated as a backend into other frameworks. There is a simple example to show the process how to build Neural Engine from source as submodule.  
Actually, the nlp_executor.cc and the CMakeLists.txt under neural_engine folder are showed how to use C++ Neural Engine. We just reuse them and modify the CMakeLists.txt to use Neural Engine as submodule.
```shell
mkdir engine_integration && cd engine_integration
git init
git submodule add https://github.com/intel/intel-extension-for-transformers itrex
git submodule update --init --recursive
cp itrex/intel_extension_for_transformers/backends/neural_engine/CMakeLists.txt .
cp itrex/intel_extension_for_transformers/backends/neural_engine/executor/src/nlp_executor.cc neural_engine_example.cc
```
Modify the NE_ROOT in the CmakeLists.txt.
```cmake
set(NE_ROOT "${PROJECT_SOURCE_DIR}/itrex/intel_extension_for_transformers/backends/neural_engine")
```

Compile neural_engine_example.cc as binary named neural_engine_example and link Nerual Engine include/lib into neural_engine_example.
```cmake
# build neural_engine_example
set(RUNTIME_OUTPUT_DIRECTORY, ${PROJECT_SOURCE_DIR})
add_executable(neural_engine_example
    neural_engine_example.cc
)

target_include_directories(neural_engine_example
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${BOOST_INC_DIRS}
)

target_link_libraries(neural_engine_example
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        gflags
        neural_engine
)
# put the neural_engine_example binary into the source dir
set_target_properties(neural_engine_example
        PROPERTIES OUTPUT_NAME neural_engine_example
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
```

Build and run the neural_engine_example.
```shell
mkdir build && cd build
cmake ..
make -j
cd ..
./neural_engine_example --config=<path to yaml file> --weight=<path to bin file> --batch_size=32 --seq_len=128 --iterations=10 --w=5
```
>**Note** Use `numactl` to bind cores when doing inference for better performance.
