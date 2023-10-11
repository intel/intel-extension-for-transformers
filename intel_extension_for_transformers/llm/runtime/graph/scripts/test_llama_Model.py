import torch
import os
import numpy as np
import struct
from transformers import AutoTokenizer, TextStreamer
from transformers import AutoModelForCausalLM
import json
from neural_compressor.adaptor.torch_utils.weight_only import quant_weight, quant_weight_w_scale
import intel_extension_for_transformers.llm.runtime.graph.chatglm2_cpp as cpp_model
GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32
GGML_QK5_0 = 32
GGML_QK5_1 = 32

def quantize_q4_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_0 in ggml.c
    assert tensor.shape[1] % GGML_QK4_0 == 0
    tensor = tensor.view(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale + 8).round().clamp(min=0, max=15).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor

def fetch_module(model, op_name):
    """Get module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.

    Returns:
        module (object).
    """
    module = model
    name_list = op_name.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    return module

def extract_gptq(model, k, v):
    print(f"Compressing {k}")
    if v["dtype"] == "fp32":
        return
    else:
        dtype = v["dtype"]
        num_bits = v["bits"]
        group_size = v["group_size"]
        scheme = v["scheme"]
    m = fetch_module(model, k)
    gptq_conf = gptq_config[k]
    if "perm" in gptq_conf:
        gptq_perm = torch.tensor(gptq_conf["perm"])
        fp32_weight = m.weight.data[:, gptq_perm]
    else:
        fp32_weight = m.weight.data
        gptq_perm = None
    gptq_scale = torch.tensor(gptq_conf["scale"])
    gptq_zp = None if scheme == "sym" else torch.tensor(gptq_conf["zero"])
    int_weight = quant_weight_w_scale(fp32_weight, gptq_scale, gptq_zp, group_size)
    return int_weight.to(torch.int8), gptq_scale, gptq_zp

def convert_transformers_to_orig(model):
    out = {}
    if "model.embed_tokens.weight" in model:
        out["tok_embeddings.weight"] = model["model.embed_tokens.weight"]
    if "model.norm.weight" in model:
        out["norm.weight"] = model["model.norm.weight"]
    if "lm_head.weight" in model:
        out["output.weight"] = model["lm_head.weight"]
    import itertools
    for i in itertools.count():
        if f"model.layers.{i}.self_attn.q_proj.weight" not in model:
            break
        out[f"layers.{i}.attention.wq.weight"] = model[f"model.layers.{i}.self_attn.q_proj.weight"]
        out[f"layers.{i}.attention.wk.weight"] = model[f"model.layers.{i}.self_attn.k_proj.weight"]
        out[f"layers.{i}.attention.wv.weight"] = model[f"model.layers.{i}.self_attn.v_proj.weight"]
        out[f"layers.{i}.attention.wo.weight"] = model[f"model.layers.{i}.self_attn.o_proj.weight"]

        out[f"layers.{i}.feed_forward.w1.weight"] = model[f"model.layers.{i}.mlp.gate_proj.weight"]
        out[f"layers.{i}.feed_forward.w2.weight"] = model[f"model.layers.{i}.mlp.down_proj.weight"]
        out[f"layers.{i}.feed_forward.w3.weight"] = model[f"model.layers.{i}.mlp.up_proj.weight"]

        out[f"layers.{i}.attention_norm.weight"] = model[f"model.layers.{i}.input_layernorm.weight"]
        out[f"layers.{i}.ffn_norm.weight"] = model[f"model.layers.{i}.post_attention_layernorm.weight"]
    return out

def replace_name(name):
    if name == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    if name == "model.norm.weight":
        return "norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    if "self_attn" in name:
        name = name.replace("model.layers", "layers")
        name = name.replace("self_attn", "attention")
        if "q_proj" in name:
            name = name.replace("q_proj", "wq")
        if "k_proj" in name:
            name = name.replace("k_proj", "wk")
        if "v_proj" in name:
            name = name.replace("v_proj", "wv")
        if "o_proj" in name:
            name = name.replace("o_proj", "wo")
        return name
    if "mlp" in name:
        name = name.replace("model.layers", "layers")
        name = name.replace("mlp", "feed_forward")
        if "gate_proj" in name:
            name = name.replace("gate_proj", "w1")
        if "down_proj" in name:
            name = name.replace("down_proj", "w2")
        if "up_proj" in name:
            name = name.replace("up_proj", "w3")
        return name
    if "input_layernorm" in name:
        name = name.replace("model.layers", "layers")
        name = name.replace("input_layernorm", "attention_norm")
        return name
    if "post_attention_layernorm" in name:
        name = name.replace("model.layers", "layers")
        name = name.replace("post_attention_layernorm", "ffn_norm")
        return name            

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():

    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1

    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))

model_name = "/mnt/disk1/data2/zhenweil/models/llama/Llama-2-7b-chat-hf"
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
from neural_compressor.utils.pytorch import load
import copy
new_model = load("/mnt/disk1/data2/zhenweil/models/llama/llama2-gptq", copy.deepcopy(model), weight_only=True)
out1 = new_model(inputs)
from neural_compressor.model import Model as INCModel
inc_model = INCModel(new_model)
qweight_config_path="/mnt/disk1/data2/zhenweil/models/llama/llama2-gptq/qconfig.json"
gptq_config_path="/mnt/disk1/data2/zhenweil/models/llama/llama2-gptq/gptq_config.json"
inc_model.export_compressed_model(
    qweight_config_path=qweight_config_path,
    gptq_config_path=gptq_config_path,
)

# import logger

with open(qweight_config_path, "r") as f:
    weight_config = json.load(f)
with open(gptq_config_path, "r") as f:
    gptq_config = json.load(f)


# list_vars = convert_transformers_to_orig(model.state_dict())
# import pdb; pdb.set_trace()
# weight_config = convert_transformers_to_orig(weight_config)
list_vars = model.state_dict()
f = open("llama_gptq_fp32.bin", "wb")

# 1. write head and params
hparams = model.config.to_dict()
f.write(b"ggjt"[::-1])  # magic
values = [
    1,  # file version
    hparams["vocab_size"],
    hparams["hidden_size"],
    256, #hparams.n_mult,
    hparams["num_attention_heads"],
    hparams["num_key_value_heads"], # n_head_kv (multi_query attention)
    hparams["num_hidden_layers"],
    hparams["hidden_size"] // hparams["num_attention_heads"],  # rot (obsolete)
    0, #file_type.value, # TODO
]
f.write(struct.pack("i" * len(values), *values))
f.write(struct.pack("i", 0))
f.write(struct.pack("f", 0))
f.write(struct.pack("f", 0))
f.write(struct.pack("i", 0))
f.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
f.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

f.write(struct.pack("i", 0))
f.write(struct.pack("i", hparams["intermediate_size"]))
f.write(struct.pack("i", 0))

f.write(struct.pack("i", 1)) # TODO, bos_token_id = 0 in https://huggingface.co/decapoda-research/llama-7b-hf/blob/main/config.json but bos_token_id = 1 in llama.cpp
f.write(struct.pack("i", 2))

f.write(struct.pack("i", 0))
f.write(struct.pack("i", 0))

# 2. vocab

encoder = tokenizer.vocab
# Add added_tokens (special tokens) to the encoder
encoder.update(tokenizer.get_added_vocab())

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

counter = 0
# sort by value
for key in sorted(encoder, key=encoder.get):
    # workaround for key error when c not found
    text=""
    for c in key:
        if c not in byte_decoder:
            text += c
        else:
            text += chr(byte_decoder[c] )
    text = bytearray( text, encoding="utf-8" )
    f.write(struct.pack("i", len(text)))
    f.write(text)
    f.write(struct.pack("f", 0))
    counter += 1

# 3. write tensors
for name in list_vars.keys():
    print(name, list_vars[name].shape, list_vars[name].dtype)
    ftype_cur = 0
    # if ".weight" in name and list_vars[name].dim() == 2:
    #     if name.replace(".weight", "") in weight_config and weight_config[name.replace(".weight", "")]["dtype"] != "fp32":
    #         ftype_cur = 13
    #     else:
    #         ftype_cur = 2

    sname = replace_name(name).encode('utf-8')
    shape = list_vars[name].shape
    f.write(struct.pack("iii", len(shape), len(sname), ftype_cur))
    f.write(struct.pack("i" * len(shape), *shape[::-1]))
    f.write(sname)
    f.seek((f.tell() + 31) & -32)
        

    # if ".weight" in name and list_vars[name].dim() == 2:
    #     # to quantize
    #     k = name.replace(".weight", "")
    #     if k in weight_config and weight_config[k]["dtype"] != "fp32":
    #         # import pdb; pdb.set_trace()
    #         # if k == "model.layers.0.mlp.down_proj":
    #         #     import pdb; pdb.set_trace()
    #         # int_weight, gptq_scale, gptq_zp = extract_gptq(model, k, weight_config[k])
    #         # dst = np.zeros((tensor.shape[0], tensor.shape[1]*2), dtype=np.int8)
    #         # byte_size = cpp_model.Model.np_jblas_qpack(np.left_shift(int_weight, 4), gptq_scale, dst)

    #         # dst = dst.flatten()
    #         # f.write(struct.pack('b' * byte_size, *list(dst[:byte_size])))
    #         m = fetch_module(new_model, k)
    #         m_weight = m.recover()
    #         m_weight.detach().numpy().tofile(f)
    #     else:
    #         print(f"q4_0 {k}")
    #         tensor = quantize_q4_0(list_vars[name])
    #         tensor.numpy().tofile(f)
    # else:
    #     # keep float32
    print(f"float {name}")
    list_vars[name].numpy().tofile(f)

f.close()