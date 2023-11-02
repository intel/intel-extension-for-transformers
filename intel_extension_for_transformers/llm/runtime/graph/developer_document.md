# Enable graph cpp model process
<img src="imgs/Enable_cpp_model.PNG" width=1200 height=150 alt="Enable cpp model">
<br>

# 1.	Model conversion
## 1.1.	Hyperparamters
The term **"hyperparamters"** describes a value that is used to configure the behavior of a large language model; this is in contrast to the model's parameters, which are the weight that were derived in the training process that was used to create the model. Each model defines its own hyperparameter structure that defines the hyperparameter values accepted by that model. Valid ITREX graph files must list these values in the correct order, and each value must be represented using the correct data type. Although hyperparameters are different across models, some attributes appear in the hyperparameters for most models:
- n_vocab: the size of the model's vocabulary
- n_embd: the size of the model's " embedding layer", which is used during prompt ingestion.
- n_layer: the number of layers in the model; each layer represents a set of weights.
Here we will use [convert_gptneox.py](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/scripts/convert_gptneox.py#L96) as an example,
```python
fout.write(struct.pack("i", hparams["num_attention_heads"]))
fout.write(struct.pack("i", hparams.get("n_head_kv", 0)))  # multi-query attention
fout.write(struct.pack("i", hparams["num_hidden_layers"]))
```
The above fout is the file we need to get, and here the num_attention, n_head_kv, and num_hidden_layer from hparams is written into fout.
## 1.2.	Vocabulary
As the name implies, a model's vocabulary comprises components that are used by the model to generate language (text). However, unlike the vocabulary of a human, which consists of words, the vocabulary of a large language model consists of "tokens". A token can be an entire word, but oftentimes they are word fragments. Just like humans can compose millions of words from just a dozen or two letters, large language models use tokens to express a large number of words from a relatively smaller number of components. Consider a vocabulary with the following tokens: `whi`, `ch`, `le`, `who`, and `a`; this vocabulary can be used to create the English words `"which"`, `"while"`, `"who"`, `"a"`, and `"leach"`. How would the behavior change if the model contained the following tokens: `wh`, `ich`, `ile`, `o`, and `leach`? Choices such as these allow model-creators to tune the behavior and performance of their models.

As described above, the model's hyperparameters typically contains a value that specifies the number of tokens in the vocabulary. The vocabulary is encoded as a list of tokens, each of which includes a 32-bit integer that specifies the length of the token.If your model has some new tokenizers, we suggest using python tokenizer from transformers and feed the input_ids to model python Api (python example in scripts floder).
Here we will use [convert_gptneox.py](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/scripts/convert_gptneox.py#L122) as an example to processed the vocabulary of gptneox and written it into fout.
```python
encoder = tokenizer.vocab
encoder.update(tokenizer.get_added_vocab())
byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}
```
## 1.3.	Model weights
The final, and largest, component of a ITREX GRAPH file is the weights of the LLM that the file represents. Abstractly, a large language model is software that is used to generate language - just like software that is used to generate images can be improved by increasing the number of colors with which images can be rendered, large language models can be improved by increasing the number of weights in the model. The total number of weights in a model are referred to as the "size" of that model. For example, the dolly-v2-3b implementation of the gpt-neox-20B language model architecture is available in a number of sizes, like 3B and 20B, which stands for 3 billion and 20 billion, respectively. These numbers refer to the total number of weights in that model. As described in the hyperparameters section, weights are grouped together in sets called "layers", which, like hyperparameters, have structures that are uniquely defined by the model architecture; within a layer, weights are grouped together in structures called "tensors". So, for instance, both dolly-v2-3B and gpt-neox-20B use layers that comprise the same tensors, but dolly-v2-3B has relatively fewer layers when compared to gpt-neox-20B.
Here we will use [convert_gptneox.py](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/scripts/convert_gptneox.py#L149) as an example to convert model weights to fout.
```python
fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
for i in range(n_dims):
fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
fout.write(str)
data.tofile(fout)
```
# 2.	Model enablements
## 2.1.	Model loading
- Model type: Refers to the type of the model, This can be compared to the model type in the Transformers library, we can see model_class in [model_type.h](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/model_utils/model_types.h#L68), here defines the basic properties of an ITREX graph model,include model_hparams, model_layer, model_struct.etc. If you has new cpp model you should update [model_archs](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/model_utils/model_types.h#L68) and [model_name_to_arch()](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/model_utils/model_types.h#L395).
- Set buffer size: You need to set the corresponding buffer size in model.h according to the size of parameters for the model, just like [gptneox.h](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/gptneox/gptneox.h), you should update [enum gptneox_model](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/gptneox/gptneox.h#L21), [model_scratch](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/gptneox/gptneox.h#L26) and [model class](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/gptneox/gptneox.h#L39).
- Model_load_internal: This function include model init and model load, The [model init function](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/gptneox/gptneox_utils.cpp#L42) initializes the model's hyperparameter, such as n_layer and n_embd parameters. 
```cpp
n_embd = hparams.n_embd;
n_vocab = hparams.n_vocab;
n_layer = hparams.n_layer;
```
The weights of the model in the ITREX Graph file will be loaded in [model load function](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/gptneox/gptneox_utils.cpp#L71). Here, we'll re-read some of the parameters and weights of the converted binary,include ffn, attention, and norm weight and bias, We'll use the mapping between the name and the weight to read the weight we need. It is shown below.
```cpp
model.others[0] = ml->get_tensor("gpt_neox.embed_in.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);
model.others[1] = ml->get_tensor("gpt_neox.final_layer_norm.weight", {n_embd}, NE_BACKEND_CPU);
model.others[2] = ml->get_tensor("gpt_neox.final_layer_norm.bias", {n_embd}, NE_BACKEND_CPU);
model.others[3] = ml->get_tensor("embed_out.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);
```
Here we use get_tensor function to read gpt_neox_embed_in.weight with a shape of (n_vocab,n_embd) tensor into model.others[0].
## 2.2.	Inference process
- Model_eval_internal: This function can be equivalent to the forward process in pytorch, which has the same computational process. In [gptneox.cpp](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/gptneox/gptneox.cpp), the model_eval_internal here will perform a complete operation on the input values, such as ffn, layernorm, mha, etc. Here's a layernorm operation:
```cpp
cur = ne_norm(ctx0, inpL);
cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[0], cur), cur),
ne_repeat(ctx0, model.layers[il].norm[1], cur));
```
It is equivalent to in [gptneox.modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L441C12-L441C12):
```python
self.input_layernorm(hidden_states)
```
The inpL in the code above is equivalent to the hidden_states in the pytorch code, and we combine ne_norm, ne_add, and ne_mul to equivalentize self.input_layernorm.
## 2.3.	Application
- Q4_0 quant : We can quantize the model generated by convert by adding a quant layer class to quantize it into an int4 low-bit file, so as to obtain better inference performance. Register quant layer class in your new_model_utils.cpp, just like [gptneox_utils.cpp](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/gptneox/gptneox_utils.cpp#L163), replace `gptneox_quant_layer` to your `new_model_quant_layer`.
- Add new CMakeList.txt: You need to add the newly added model to the following CMakeList.txt. New model CMakeList.txt just like [gptneox_CMakeList.txt](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/gptneox/CMakeLists.txt), and [models_CMakeList.txt](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/models/CMakeLists.txt).

### Python bindings and scripts
1. Modify codes for python API

files need to be modified:
- `intel_extension_for_transformers/llm/runtime/graph/application/CMakeLists.txt`
- `intel_extension_for_transformers/llm/runtime/graph/application/main_pybind.cpp`
- `intel_extension_for_transformers/llm/runtime/graph/__init__.py`

If `new_model` will be added, modify the code as follows:
```diff
diff --git a/intel_extension_for_transformers/llm/runtime/graph/__init__.py b/intel_extension_for_transformers/llm/runtime/graph/__init__.py
index aaeab8d16a..12a835e652 100644
--- a/intel_extension_for_transformers/llm/runtime/graph/__init__.py
+++ b/intel_extension_for_transformers/llm/runtime/graph/__init__.py
@@ -57,6 +57,8 @@ class Model:
             import intel_extension_for_transformers.llm.runtime.graph.baichuan_cpp as cpp_model
         elif model_name == "polyglot":
             import intel_extension_for_transformers.llm.runtime.graph.polyglot_cpp as cpp_model
+        elif model_name == "new_model": # read from config.json->model_type
+            import intel_extension_for_transformers.llm.runtime.graph.new_model_cpp as cpp_model
         else:
             raise TypeError("Unspported model type {}!".format(model_name))
         self.module = cpp_model
diff --git a/intel_extension_for_transformers/llm/runtime/graph/application/CMakeLists.txt b/intel_extension_for_transformers/llm/runtime/graph/application/CMakeLists.txt
index d86107d26e..36d30cabe3 100644
--- a/intel_extension_for_transformers/llm/runtime/graph/application/CMakeLists.txt
+++ b/intel_extension_for_transformers/llm/runtime/graph/application/CMakeLists.txt
@@ -67,6 +67,7 @@ compile_quant(quant_chatglm   quant_model.cpp chatglm   chatglm)
 compile_quant(quant_chatglm2  quant_model.cpp chatglm2  chatglm2)
 compile_quant(quant_baichuan  quant_model.cpp baichuan  baichuan)
 compile_quant(quant_mistral   quant_model.cpp mistral   llama)
+compile_quant(quant_new_model   quant_model.cpp new_model   new_model)
 
 # all models running
 if (NE_PYTHON_API)
@@ -88,6 +89,7 @@ set(mymap_chatglm 11)
 set(mymap_baichuan 12)
 set(mymap_polyglot 13)
 set(mymap_mistral 14)
+set(mymap_new_model 15)
 
 function(compile_run TARGET SRC MODEL_NAME MODEL_LIB)
  add_executable_w_warning(${TARGET} ${SRC})
@@ -120,3 +122,4 @@ compile_run(run_chatglm2  main_run.cpp chatglm2  chatglm2)
 compile_run(run_chatglm   main_run.cpp chatglm   chatglm)
 compile_run(run_baichuan  main_run.cpp baichuan  baichuan)
 compile_run(run_mistral   main_run.cpp mistral   llama)
+compile_run(run_new_model   main_run.cpp new_model   new_model)
diff --git a/intel_extension_for_transformers/llm/runtime/graph/application/main_pybind.cpp b/intel_extension_for_transformers/llm/runtime/graph/application/main_pybind.cpp
index 894be0134d..a9a57c0a9e 100644
--- a/intel_extension_for_transformers/llm/runtime/graph/application/main_pybind.cpp
+++ b/intel_extension_for_transformers/llm/runtime/graph/application/main_pybind.cpp
@@ -471,6 +471,10 @@ PYBIND11_MODULE(polyglot_cpp, m)
 
 PYBIND11_MODULE(mistral_cpp, m)
 
+#elif MODEL_NAME_ID == 15
+
+PYBIND11_MODULE(new_model_cpp, m)
+
 #endif
 {
   m.doc() = "cpp model python binding";
```


2. Use python API

[how-to-use-transformer-based-api](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/llm/runtime/graph/README.md#how-to-use-transformer-based-api)

You can use Python API to run Hugging Face model simply. Here is the sample code:

```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
model_name = "Intel/neural-chat-7b-v1-1"     # Hugging Face model_id or local model
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```


# 3.	Performance optimization 
## 3.1.	Quantize model and use Jblas library for better performance
Quantize model and use the jblas library for inference can lead to better performance
```bash

# convert the model directly use model path
python scripts/convert_new_model.py --outtype f32 --outfile ne-f32.bin new_model_path
# optimized INT4 model with group size 128 (recommended)
./build/bin/quant_new_model --model_file ne-f32.bin --out_file ne-q4_j.bin --weight_dtype int4 --group_size 128 --compute_dtype int8
```
Then you can use the model to inference according to the process in the [README](https://github.com/intel/intel-extension-for-transformers/tree/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph).
## 3.2.	MHA fusion
We can improve the performance by fusion the multihead attention process.
- [MHA-Fusion Introduction](https://github.com/intel/intel-extension-for-transformers/blob/graph_developer_document/intel_extension_for_transformers/llm/runtime/graph/fused_attention.md)
- [MHA-Fusion example](https://github.com/intel/intel-extension-for-transformers/pull/567)
## 3.3.	FFN fusion
We can improve the performance by fusion the FFN process
- [FFN-Fusion example](https://github.com/intel/intel-extension-for-transformers/pull/160)
# 4. A complete example
- [Enable baichuan](https://github.com/intel/intel-extension-for-transformers/pull/376)

