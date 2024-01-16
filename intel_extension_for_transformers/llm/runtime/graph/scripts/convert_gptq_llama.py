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

import os
import re
import argparse
from common import *


def permute_func(weights, n_head: int, n_head_kv: int):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head //= n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2,
                            *weights.shape[1:]).swapaxes(1, 2).reshape(weights.shape))

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    out_path = args.outfile.as_posix()
    model_path = args.model.as_posix()

    model, config, quantize_config = load_gptq_model(model_path)
    lm_head_state_dict = model["lm_head.weight"]
    import pdb;pdb.set_trace();
    from neural_compressor.utils.load_huggingface import export_compressed_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = export_compressed_model(model, "/mnt/data2/changwa1/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization/llama_gptq")
    model = model.state_dict()
    model["lm_head.weight"] = lm_head_state_dict
    import pdb;pdb.set_trace();
    f = open(out_path, "wb")

    # 1. write hparams
    n_vocab = config["vocab_size"]
    n_embd = config["hidden_size"]
    n_layer = config["num_hidden_layers"]
    n_head = config["num_attention_heads"]
    ffn_hidden_size = config["intermediate_size"]

    # hardcoded:
    n_mult = 256

    # 1. write head and params
    f.write(b"ggjt"[::-1])  # magic

    n_head = n_head
    n_head_kv = n_head
    values = [
        1,  # file version
        n_vocab,
        n_embd,
        256,  #hparams.n_mult,
        n_head,
        n_head_kv,  # n_head_kv (multi_query attention)
        n_layer,
        n_embd // n_head,  # rot (obsolete)
        0,  #file_type.value, # TODO
    ]
    f.write(struct.pack("i" * len(values), *values))
    f.write(struct.pack("i", 0))
    f.write(struct.pack("f", 0))
    f.write(struct.pack("f", 0))
    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    f.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", ffn_hidden_size))
    f.write(struct.pack("i", 0))

    f.write(struct.pack("f", config["rms_norm_eps"]))
    f.write(struct.pack("f", config["rope_theta"] if "rope_theta" in config else 10000))
    rope_scale = 1
    if config.get("rope_scaling") is not None:
        rope_scale = config["rope_scaling"].get("factor", 1)
    f.write(struct.pack("f", rope_scale))

    # TODO, bos_token_id = 0 in https://huggingface.co/decapoda-research/llama-7b-hf/blob/main/config.json
    # but bos_token_id = 1 in llama.cpp
    f.write(struct.pack("i", 1))  
    f.write(struct.pack("i", 2))

    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", 0))

    # 2. vocab
    tokenizer_path = os.path.join(model_path, "tokenizer.model")
    vocab = load_vocab(Path(tokenizer_path))
    for text, score in vocab.all_tokens():
        f.write(struct.pack("i", len(text)))
        f.write(text)
        f.write(struct.pack("f", score))

    # 3. write tensors
    list_vars = model
    convert_fp32_tensor("model.embed_tokens.weight", "tok_embeddings.weight", list_vars, f)
    convert_fp32_tensor("model.norm.weight", "norm.weight", list_vars, f)
    convert_fp32_tensor("lm_head.weight", "output.weight", list_vars, f)
    #convert_q4_jblas_tensor("lm_head", "output.weight", list_vars, f, quantize_config, n_head)
    for i in range(n_layer):
        convert_q4_jblas_tensor(f"model.layers.{i}.self_attn.q_proj",
                    f"layers.{i}.attention.wq.weight", list_vars, f, quantize_config, n_head, n_head,
                    permute_func=permute_func)
        convert_q4_jblas_tensor(f"model.layers.{i}.self_attn.k_proj",
                    f"layers.{i}.attention.wk.weight", list_vars, f, quantize_config, n_head, n_head_kv,
                    permute_func=permute_func)
        convert_q4_jblas_tensor(f"model.layers.{i}.self_attn.v_proj",
                    f"layers.{i}.attention.wv.weight", list_vars, f, quantize_config, n_head)
        convert_q4_jblas_tensor(f"model.layers.{i}.self_attn.o_proj",
                    f"layers.{i}.attention.wo.weight", list_vars, f, quantize_config, n_head)
        convert_q4_jblas_tensor(f"model.layers.{i}.mlp.gate_proj",
                    f"layers.{i}.feed_forward.w1.weight", list_vars, f, quantize_config, n_head)
        convert_q4_jblas_tensor(f"model.layers.{i}.mlp.down_proj",
                    f"layers.{i}.feed_forward.w2.weight", list_vars, f, quantize_config, n_head)
        convert_q4_jblas_tensor(f"model.layers.{i}.mlp.up_proj",
                    f"layers.{i}.feed_forward.w3.weight", list_vars, f, quantize_config, n_head)

        convert_fp32_tensor(f"model.layers.{i}.input_layernorm.weight",
                        f"layers.{i}.attention_norm.weight", list_vars, f)
        convert_fp32_tensor(f"model.layers.{i}.post_attention_layernorm.weight",
                        f"layers.{i}.ffn_norm.weight", list_vars, f)


    f.close()
    print(f"Success! saved as {out_path}")


if __name__ == '__main__':
    main()
