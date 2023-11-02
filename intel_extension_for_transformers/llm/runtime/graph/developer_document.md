# Enable graph cpp model process
<img src="imgs/Enable_cpp_model.PNG" width=1200 height=150 alt="Enable cpp model">
<br>

## Before you start
ITREX LLM C++ Runtime has already supported some popular models like `LLAMA`,`GPT-J`, `GPT-NEOX`, `DOLLY`, etc.These LLMs have similar architectures and some of them share the same architect (`DOLLY` and `GPT-NEOX`). Before adding a new model, you can checkout its architecture (from Huggingface `config.json`) whether is in our [supported list](https://github.com/intel/intel-extension-for-transformers/blob/1.2.1/intel_extension_for_transformers/llm/runtime/graph/models/model_utils/model_types.h#L68).

However, LLM inference thing is complicated. It may have its own: 1. special tokenizer (or vocab); 2. architecture (or forward pipeline); 3. operators (or kernels). Generally speaking, the first and second points appear frequently for transformers-LLMs. I will show you how to run a new model as soon as possible when your model hasn't any problems like above or only the problem 1.

For simplicity, we take [polyglot](https://huggingface.co/EleutherAI/polyglot-ko-5.8b) as the example model. It has the same architecture as `GPT-NEOX` but only fewer layers.

Firstly, we need to add its temp buffer in its [related model-arch header file](https://github.com/intel/intel-extension-for-transformers/blob/1.2.1/intel_extension_for_transformers/llm/runtime/graph/models/gptneox/gptneox.h) and [re-compile](https://github.com/intel/intel-extension-for-transformers/blob/1.2.1/intel_extension_for_transformers/llm/runtime/graph/README.md#1-install-llm-runtime).
```diff
static const model_scratch gptneox_mem_req(int n_layers) {
  switch (n_layers) {
    case 44:
      return {2048ull * MB, 2048ull * MB, 4096ull * MB};
    case 32:
      return {512ull * MB, 512ull * MB, 1026ull * MB};
+   case 28:  // 5.8B
+     return {512ull * MB, 512ull * MB, 1024ull * MB};
    default:
      MODEL_ASSERT(false);
  }
}
```

Secondly, convert its PyTorch FP32 weights into our format ([reference section](https://github.com/intel/intel-extension-for-transformers/blob/1.2.1/intel_extension_for_transformers/llm/runtime/graph/README.md#1-convert-and-quantize-llm)). 

command:
```bash
python scripts/convert.py --outtype f32 --outfile ne-f32.bin EleutherAI/polyglot-ko-5.8b
```

Finally, use `transformers` tokenizer to encode prompt and decode return tokens instead of re-implementing C++ tokenizer.

For checking text generation results, we recommend you to run this naive python codes below to align our runtime engine outputs with PyTorch (`FP32 data type, greedy search`).
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
from intel_extension_for_transformers.llm.runtime.graph import Model

model_name = "EleutherAI/polyglot-ko-5.8b"
prompt = "she open the door and see"
# prompt = "옛날 옛적에 어린 소녀가 있었어요"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids

# pt infer
pt_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
pt_model.eval()
pt_outputs = pt_model.generate(inputs, do_sample=False, max_new_tokens=128)
pt_ans = tokenizer.batch_decode(pt_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("=====pytorch result======")
print(pt_ans)

# itrex infer
model = Model()
model.init_from_bin("polyglot", "ne-f32.bin", do_sample=False, max_new_tokens=128)
outputs = model.generate(inputs, do_sample=False, max_new_tokens=128)
ans = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("=====itrex result======")
print(ans)
```

The English prompt would have the output like:
```bash
=====pytorch result======
she open the door and see him. She looks at him and says, "How do you do?" He says, "Fine." She says, "What do you want?" He says, "I want to go home." She says, "Where are you going?" He says, "I'm going home." She says, "Where are you going?" He says, "I'm

=====itrex result======
she open the door and see him. She looks at him and says, "How do you do?" He says, "Fine." She says, "What do you want?" He says, "I want to go home." She says, "Where are you going?" He says, "I'm going home." She says, "Where are you going?" He says, "I'm
```

The Korean prompt would have the output like:
```bash
=====pytorch result======
옛날 옛적에 어린 소녀가 있었어요. 그 소녀는 어느 날, 숲 속에서 길을 잃고 헤매다가 한 마리의 동물을 만나게 되었어요. 그 동물은 소녀에게 길을 안내해 주겠다고 하였어요. 소녀는 그 동물을 따라 숲 속으로 들어갔어요. 한참을 걷고 있는데, 갑자기 동물이 소녀를 땅 속으로 끌고 들어가는 것이었어요. 소녀는 깜짝 놀라서 소리쳤어요. "안돼! 나를 죽이려고 하는 거야?" 그러자 동물은 소녀에게 조용히 말했어요. "쉿! 조용히 해. 내 말을 잘 들어봐. 저 앞에

=====itrex result======
옛날 옛적에 어린 소녀가 있었어요. 그 소녀는 어느 날, 숲 속에서 길을 잃고 헤매다가 한 마리의 동물을 만나게 되었어요. 그 동물은 소녀에게 길을 안내해 주겠다고 하였어요. 소녀는 그 동물을 따라 숲 속으로 들어갔어요. 한참을 걷고 있는데, 갑자기 동물이 소녀를 땅 속으로 끌고 들어가는 것이었어요. 소녀는 깜짝 놀라서 소리쳤어요. "안돼! 나를 죽이려고 하는 거야?" 그러자 동물은 소녀에게 조용히 말했어요. "쉿! 조용히 해. 내 말을 잘 들어봐. 저 앞에
```

Once you make sure your model has the same generated tokens as PyTorch, you can deploy it by using more `transformers` style python codes and `INT4` type. Please refer to `Python API` section for more details.


# 1.	Model conversion
## 1.1.	Hyperparamters
The term **"hyperparamters"** describes a value that is used to configure the behavior of a large language model; this is in contrast to the model's parameters, which are the weight that were derived in the training process that was used to create the model. Each model defines its own hyperparameter structure that defines the hyperparameter values accepted by that model. Valid ITREX graph files must list these values in the correct order, and each value must be represented using the correct data type. Although hyperparameters are different across models, some attributes appear in the hyperparameters for most models:
- n_vocab: the size of the model's vocabulary
- n_embd: the size of the model's " embedding layer", which is used during prompt ingestion.
- n_layer: the number of layers in the model; each layer represents a set of weights.
Here we will use convert_gptneox.py as an example,
```python
fout.write(struct.pack("i", hparams["num_attention_heads"]))
fout.write(struct.pack("i", hparams.get("n_head_kv", 0)))  # multi-query attention
fout.write(struct.pack("i", hparams["num_hidden_layers"]))
```
The above fout is the file we need to get, and here the num_attention, n_head_kv, and num_hidden_layer from hparams is written into fout.
## 1.2.	Vocabulary
As the name implies, a model's vocabulary comprises components that are used by the model to generate language (text). However, unlike the vocabulary of a human, which consists of words, the vocabulary of a large language model consists of "tokens". A token can be an entire word, but oftentimes they are word fragments. Just like humans can compose millions of words from just a dozen or two letters, large language models use tokens to express a large number of words from a relatively smaller number of components. Consider a vocabulary with the following tokens: whi, ch le, who, and a; this vocabulary can be used to create the English words "which", "while", "who", "a", and "leach". How would the behavior change if the model contained the following tokens: wh, ich, ile, o, and leach? Choices such as these allow model-creators to tune the behavior and performance of their models.
As described above, the model's hyperparameters typically contains a value that specifies the number of tokens in the vocabulary. The vocabulary is encoded as a list of tokens, each of which includes a 32-bit integer that specifies the length of the token.If your model has some new tokenizers, we suggest using python tokenizer from transformers and feed the input_ids to model python Api (python example in scripts floder).
Here we will use convert_gptneox.py as an example to processed the vocabulary of gptneox and written it into fout.
```python
encoder = tokenizer.vocab
encoder.update(tokenizer.get_added_vocab())
byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}
```
## 1.3.	Model weights
The final, and largest, component of a ITREX GRAPH file is the weights of the LLM that the file represents. Abstractly, a large language model is software that is used to generate language - just like software that is used to generate images can be improved by increasing the number of colors with which images can be rendered, large language models can be improved by increasing the number of weights in the model. The total number of weights in a model are referred to as the "size" of that model. For example, the dolly-v2-3b implementation of the gpt-neox-20B language model architecture is available in a number of sizes, like 3B and 20B, which stands for 3 billion and 20 billion, respectively. These numbers refer to the total number of weights in that model. As described in the hyperparameters section, weights are grouped together in sets called "layers", which, like hyperparameters, have structures that are uniquely defined by the model architecture; within a layer, weights are grouped together in structures called "tensors". So, for instance, both dolly-v2-3B and gpt-neox-20B use layers that comprise the same tensors, but dolly-v2-3B has relatively fewer layers when compared to gpt-neox-20B.
Here we will use convert_gptneox.py as an example to convert model weights to fout.
```python
fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
for i in range(n_dims):
fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
fout.write(str)
data.tofile(fout)
```
# 2.	Model enablements
## 2.1.	Model loading
- Model type: Refers to the type of the model, This can be compared to the model type in the Transformers library, we can see model_class in model_type.h, here defines the basic properties of an ITREX graph model,include model_hparams, model_layer, model_struct.etc. If you has new cpp model you should update the table.
- Model struct:
- Model layer:
- Set buffer size: We need to set the corresponding buffer size in model.h according to the n_layers of the model.(not n_layer)
- Model_load_internal: This function include model init and model load, The model init function initializes the model's hyperparameter, such as n_layer and n_embd parameters. 
```cpp
n_embd = hparams.n_embd;
n_vocab = hparams.n_vocab;
n_layer = hparams.n_layer;
```
The weights of the model in the ITREX Graph file will be loaded in model load function. Here, we'll re-read some of the parameters and weights of the converted binary,include ffn, attention, and norm weight and bias, We'll use the mapping between the name and the weight to read the weight we need. It is shown below.
```cpp
model.others[0] = ml->get_tensor("gpt_neox.embed_in.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);
```
Here we use get_tensor function to read gpt_neox_embed_in.weight with a shape of (n_vocab,n_embd) tensor into model.others[0].
## 2.2.	Inference process
- Model_eval_internal: This function can be equivalent to the forward process in pytorch, which has the same computational process. In gptneox.cpp, the model_eval_internal here will perform a complete operation on the input values, such as ffn, layernorm, mha, etc. Here's a layernorm operation:
```cpp
cur = ne_norm(ctx0, inpL);
cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[0], cur), cur),
ne_repeat(ctx0, model.layers[il].norm[1], cur));
```
It is equivalent to in gptneox.modeling:(KV_cache diff)(multi batch)
```python
self.input_layernorm(hidden_states)
```
The inpL in the code above is equivalent to the hidden_states in the pytorch code, and we combine ne_norm, ne_add, and ne_mul to equivalentize self.input_layernorm.
## 2.3.	Application
- Q4_0 quant : We can quantize the model generated by convert by adding a quant layer class to quantize it into an int4 low-bit file, so as to obtain better inference performance. Register quant layer class in your model_utils.cpp, just like gptneox_utils.cpp, commands
- Add the model to the Cmake file: We need to add the newly added model to the following CMakeList.txt.1,2,3,

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
## 3.1.	Introduction to JBLAS,isa
## 3.2.	Int4 (q4_j) quantize (sym,bits,compute type, fallback)
- Bits :
- Compute type
## 3.3.	MHA fusion
- MHA-Fusion Introduction
## 3.4.	FFN fusion
- Link-to-FFN-doc
# 4.	Example to Enable baichuan

