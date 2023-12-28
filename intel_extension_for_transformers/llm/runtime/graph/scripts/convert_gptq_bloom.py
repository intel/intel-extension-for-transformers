#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import numpy as np
import struct
from transformers import AutoTokenizer, TextStreamer, AutoConfig
from transformers import AutoModelForCausalLM
import json
import copy
from neural_compressor.adaptor.torch_utils.weight_only import quant_weight, quant_weight_w_scale
import intel_extension_for_transformers.llm.runtime.graph.chatglm2_cpp as cpp_model

GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32
GGML_QK5_0 = 32
GGML_QK5_1 = 32


def quantize_q4_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_0 in ggml.c
    # import pudb; pudb.set_trace()
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
    m_weight = m.recover()
    # import pdb; pdb.set_trace()
    gptq_conf = gptq_config[k]
    if "perm" in gptq_conf:
        gptq_perm = torch.tensor(gptq_conf["perm"])
        fp32_weight = m_weight[:, gptq_perm]
    else:
        fp32_weight = m_weight
        gptq_perm = None
    gptq_scale = torch.tensor(gptq_conf["scale"])
    gptq_zp = None if scheme == "sym" else torch.tensor(gptq_conf["zero"])
    int_weight = quant_weight_w_scale(fp32_weight, gptq_scale, gptq_zp, group_size)
    return int_weight.to(torch.int8), gptq_scale, gptq_zp


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
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1

    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))


model_name = "/mnt/disk1/data2/zhenweil/models/bloom/bloom-7b1"
prompt = "Once upon a time, a little girl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

gptq_model = "/mnt/disk1/data2/zhenweil/models/bloom/bloom-gptq/"
from neural_compressor.utils.pytorch import load

new_model = load(gptq_model, copy.deepcopy(model), weight_only=True)
new_model_bk = copy.deepcopy(new_model)
from neural_compressor.model import Model as INCModel

inc_model = INCModel(new_model)
qweight_config_path = gptq_model + "qconfig.json"
gptq_config_path = gptq_model + "gptq_config.json"
inc_model.export_compressed_model(qweight_config_path=qweight_config_path, gptq_config_path=gptq_config_path)

with open(qweight_config_path, "r") as f:
    weight_config = json.load(f)
with open(gptq_config_path, "r") as f:
    gptq_config = json.load(f)

list_vars = new_model_bk.state_dict()
f = open("bloom_gptq_q4.bin", "wb")

# 1. write head and params
hparams = config.to_dict()
ftype = 0
f.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex

f.write(struct.pack("i", hparams["vocab_size"]))
f.write(struct.pack("i", hparams["hidden_size"]))
f.write(struct.pack("i", 1))
f.write(struct.pack("i", hparams["n_head"]))
f.write(struct.pack("i", hparams.get("n_head_kv", 0)))  # multi-query attention
f.write(struct.pack("i", hparams["n_layer"]))
f.write(struct.pack("i", 0))
f.write(struct.pack("i", ftype))
f.write(struct.pack("i", 0))
f.write(struct.pack("f", 0))
f.write(struct.pack("f", 0))
f.write(struct.pack("i", 0))
f.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
f.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

f.write(struct.pack("i", 0))
f.write(struct.pack("i", 0))
f.write(struct.pack("i", 0))
fout.write(struct.pack("f", 1e-6))  # rms norm eps

f.write(struct.pack("i", tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1))
f.write(struct.pack("i", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2))
f.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1))
f.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))

# 2. vocab
reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}
byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

for i in range(hparams["vocab_size"]):
    text = tokenizer.decode([i]).encode('utf-8')
    f.write(struct.pack("i", len(text)))
    f.write(text)

# 3. write tensors
for name in list_vars.keys():
    src = name
    if "query_key_value" in src:
        q_d, k_d, v_d = list_vars[src].reshape(config.n_head, 3, -1).unbind(1)
        list_vars[src] = torch.cat([q_d, k_d, v_d], dim=0).reshape_as(list_vars[src])

    ftype_cur = 0
    if ".weight" in name and list_vars[name].dim() == 2:
        ftype_cur = 2  # TODO(Zhenwei) support jblas
    if list_vars[src].dtype == "torch.bfloat16":
        list_vars[src]=list_vars[src].float()
    data = list_vars[src].squeeze().numpy()
    data = data.astype(np.float32)

    n_dims = len(data.shape)
    print(name, n_dims, data.shape)
    str = name.encode('utf-8')
    f.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        f.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    f.write(str)

    if ".weight" in name and list_vars[name].dim() == 2:
        # to quantize
        k = name.replace(".weight", "")
        if k in weight_config and weight_config[k]["dtype"] != "fp32":
            print(f"jblas {k}")
            int_weight, gptq_scale, gptq_zp = extract_gptq(new_model, k, weight_config[k])

            tensor = int_weight.view(-1, 32) + 8
            tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
            gptq_scale = gptq_scale.view(-1, 1)
            gptq_scale = torch.cat([gptq_scale, gptq_scale, gptq_scale, gptq_scale], dim=1).view(-1, 1)
            tensor = torch.cat((gptq_scale.half().view(torch.int8), tensor), dim=-1)
            if "query_key_value" in src:
                q_d, k_d, v_d = tensor.reshape(config.n_head, 3, -1).unbind(1)
                tensor = torch.cat([q_d, k_d, v_d], dim=0).reshape_as(tensor)
            tensor.numpy().tofile(f)

        else:
            print(f"q4_0 {k}")
            tensor = quantize_q4_0(list_vars[name])
            tensor.numpy().tofile(f)
    else:
        # keep float32
        print(f"float {name}")
        data.tofile(f)
    # break
f.close()
