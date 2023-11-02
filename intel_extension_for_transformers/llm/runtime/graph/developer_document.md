# 1.	Model conversion
## 1.1.	Hyperparamters
The term "hyperparamters" describes a value that is used to configure the behavior of a large language model; this is in contrast to the model's parameters, which are the weight that were derived in the training process that was used to create the model. Each model defines its own hyperparameter structure that defines the hyperparameter values accepted by that model. Valid ITREX graph files must list these values in the correct order, and each value must be represented using the correct data type. Although hyperparameters are different across models, some attributes appear in the hyperparameters for most models:
•	n_vocab: the size of the model's vocabulary
•	n_embd: the size of the model's " embedding layer", which is used during prompt ingestion.
•	n_layer: the number of layers in the model; each layer represents a set of weights.
Here we will use convert_gptneox.py as an example,
fout.write(struct.pack("i", hparams["num_attention_heads"]))
fout.write(struct.pack("i", hparams.get("n_head_kv", 0)))  # multi-query attention
fout.write(struct.pack("i", hparams["num_hidden_layers"]))
The above fout is the file we need to get, and here the num_attention, n_head_kv, and num_hidden_layer from hparams is written into fout.
## 1.2.	Vocabulary
As the name implies, a model's vocabulary comprises components that are used by the model to generate language (text). However, unlike the vocabulary of a human, which consists of words, the vocabulary of a large language model consists of "tokens". A token can be an entire word, but oftentimes they are word fragments. Just like humans can compose millions of words from just a dozen or two letters, large language models use tokens to express a large number of words from a relatively smaller number of components. Consider a vocabulary with the following tokens: whi, ch le, who, and a; this vocabulary can be used to create the English words "which", "while", "who", "a", and "leach". How would the behavior change if the model contained the following tokens: wh, ich, ile, o, and leach? Choices such as these allow model-creators to tune the behavior and performance of their models.
As described above, the model's hyperparameters typically contains a value that specifies the number of tokens in the vocabulary. The vocabulary is encoded as a list of tokens, each of which includes a 32-bit integer that specifies the length of the token.If your model has some new tokenizers, we suggest using python tokenizer from transformers and feed the input_ids to model python Api (python example in scripts floder).
Here we will use convert_gptneox.py as an example to processed the vocabulary of gptneox and written it into fout.
encoder = tokenizer.vocab
encoder.update(tokenizer.get_added_vocab())
byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}
## 1.3.	Model weights
The final, and largest, component of a ITREX GRAPH file is the weights of the LLM that the file represents. Abstractly, a large language model is software that is used to generate language - just like software that is used to generate images can be improved by increasing the number of colors with which images can be rendered, large language models can be improved by increasing the number of weights in the model. The total number of weights in a model are referred to as the "size" of that model. For example, the dolly-v2-3b implementation of the gpt-neox-20B language model architecture is available in a number of sizes, like 3B and 20B, which stands for 3 billion and 20 billion, respectively. These numbers refer to the total number of weights in that model. As described in the hyperparameters section, weights are grouped together in sets called "layers", which, like hyperparameters, have structures that are uniquely defined by the model architecture; within a layer, weights are grouped together in structures called "tensors". So, for instance, both dolly-v2-3B and gpt-neox-20B use layers that comprise the same tensors, but dolly-v2-3B has relatively fewer layers when compared to gpt-neox-20B.
Here we will use convert_gptneox.py as an example to convert model weights to fout.
fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
for i in range(n_dims):
fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
fout.write(str)
data.tofile(fout)
# 2.	Model enablements
## 2.1.	Model loading
•	Model type: Refers to the type of the model, This can be compared to the model type in the Transformers library, we can see model_class in model_type.h, here defines the basic properties of an ITREX graph model,include model_hparams, model_layer, model_struct.etc. If you has new cpp model you should update the table.
•	Model struct:
•	Model layer:
•	Set buffer size: We need to set the corresponding buffer size in model.h according to the n_layers of the model.(not n_layer)
•	Model_load_internal: This function include model init and model load, The model init function initializes the model's hyperparameter, such as n_layer and n_embd parameters. 
n_embd = hparams.n_embd;
n_vocab = hparams.n_vocab;
n_layer = hparams.n_layer;
•	 The weights of the model in the ITREX Graph file will be loaded in model load function. Here, we'll re-read some of the parameters and weights of the converted binary,include ffn, attention, and norm weight and bias, We'll use the mapping between the name and the weight to read the weight we need. It is shown below.
model.others[0] = ml->get_tensor("gpt_neox.embed_in.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);
Here we use get_tensor function to read gpt_neox_embed_in.weight with a shape of (n_vocab,n_embd) tensor into model.others[0].
## 2.2.	Inference process
                 Copy inputs, model forward, explain tensor layout format [a, b] ---> shape is b x a, copy logits out
•	Model_eval_internal: This function can be equivalent to the forward process in pytorch, which has the same computational process. In gptneox.cpp, the model_eval_internal here will perform a complete operation on the input values, such as ffn, layernorm, mha, etc. Here's a layernorm operation:
cur = ne_norm(ctx0, inpL);
cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[0], cur), cur),
ne_repeat(ctx0, model.layers[il].norm[1], cur));
			It is equivalent to in gptneox.modeling:(KV_cache diff)(multi batch)
self.input_layernorm(hidden_states)
The inpL in the code above is equivalent to the hidden_states in the pytorch code, and we combine ne_norm, ne_add, and ne_mul to equivalentize self.input_layernorm.
## 2.3.	Application
•	Q4_0 quant : We can quantize the model generated by convert by adding a quant layer class to quantize it into an int4 low-bit file, so as to obtain better inference performance. Register quant layer class in your model_utils.cpp, just like gptneox_utils.cpp, commands
•	Add the model to the Cmake file: We need to add the newly added model to the following CMakeList.txt.1,2,3,
•	Python bindings and scripts☹zhenwei
•	
# 3.	Performance optimization 
## 3.1.	Introduction to JBLAS,isa
## 3.2.	Int4 (q4_j) quantize (sym,bits,compute type, fallback)
•	Sym:
•	Bits :
•	Compute type
•	fallback
## 3.3.	MHA fusion
•	Basic MHA
•	GQA
•	MHA-Fusion Introduction
## 3.4.	FFN fusion
•	Basic FFN
•	Link-to-FFN-doc
# 4.	Example to Enable baichuan

