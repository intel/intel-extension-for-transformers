# Compile a TensorFlow Model to Neural Engine IR

The `Neural Engine` as a backend for `Intel® Extension for Transformers` currently supports frozen static graph models from two deep learning frameworks (`TensorFlow` and `ONNX`).

The following image shows the `Intel® Extension for Transformers` workflow to compile a framework model to its intermediate representation(IR).

1. The `Loader` loads models from different deep learning frameworks. 
2. The `Extractors` extract original model operators to compose the engine graph.
3. The `Subgraph matcher`  implements pattern fusion to accelerate inference. 
4. The `Emitter` saves the intermediate graph to the disk in the format of `.yaml` and `.bin` files.

![](imgs/compile_workflow.png)

# Example

The following example shows compiling a `TensorFlow` model to the `Neural Engine` IR. It compiles the classic NLP model `bert_base` on task `MRPC`.

## Prepare your environment

  ```shell  
  # conda create a new work environment
  conda create -n <your_env_name> python=3.7
  conda activate <your_env_name>

  # clone the intel-extension-for-transformers repository
  git clone https://github.com/intel/intel-extension-for-transformers

  # install Intel-TensorFlow
  pip install https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp37-cp37m-manylinux2010_x86_64.whl

  # install other necessary requirements
  pip install -r requirements.txt
  ```

## Compile the bert_base model to Engine IR

```python
# import neural engine compile module
from intel_extension_for_transformers.backends.neural_engine.compile import compile
# convert the graph to get the IR
graph = compile("./model/bert_base_mrpc.pb")
# the yaml and bin file will be stored in the './ir' folder by default.
graph.save('./output_dir')
```

**Note**: The `bert_base` example should come from the `Intel` model zoo, comming soon.