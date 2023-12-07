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
import json
import sys
import re
import argparse
from common import *

def permute_func(weights, n_head: int, n_head_kv: int):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head //= n_head_kv
    return (weights.reshape(n_head_kv, 2, weights.shape[0] // n_head_kv // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

# def unpack_weight(qweight, scales, qzeros, permute=False, group_size=32, bits=4):
#     wf = torch.tensor([[ 0,  4,  8, 12, 16, 20, 24, 28]], dtype=torch.int32)
#     zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int16 if bits == 8 else torch.int8)
#     torch.bitwise_and(zeros, (2 ** bits) - 1, out=zeros)
#     # import pdb; pdb.set_trace()
#     zeros = zeros + 1
#     zeros[zeros == 16] = 0
#     zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

#     scales = scales
#     scales = scales.reshape(-1, 1, scales.shape[-1])
        
#     weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
#     torch.bitwise_and(weight,(2 ** bits) - 1, out=weight)
#     int_weight = weight.reshape(-1, group_size, weight.shape[2])

#     return int_weight, scales, zeros
    # weight = (scales * (weight - zeros))
    # weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    # return weight.t(), pack_tensor

def convert_q4_tensor(src_name, dst_name, model, fout, n_head, n_head2=0, permute=False):
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    # g_idx = model[f"{src_name}.g_idx"]
    qweight = model[f"{src_name}.qweight"]
    # import pdb; pdb.set_trace()
    int_weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros, permute)
    # shape = int_weight.view(-1, int_weight.shape[-1]).shape
    # write_header(fout, shape, dst_name, 2)

    int_weight = int_weight.view(-1,int_weight.shape[-1]).t()
    gptq_scales = gptq_scales.view(-1,gptq_scales.shape[-1]).t()
    gptq_zeros = gptq_zeros.view(-1,gptq_zeros.shape[-1]).t()

    write_header(fout, int_weight.shape, dst_name, 2)
    if permute:
        int_weight = permute_func(int_weight, n_head, n_head2).contiguous()
        gptq_scales = permute_func(gptq_scales, n_head, n_head2).contiguous()
        gptq_zeros = permute_func(gptq_zeros, n_head, n_head2).contiguous()

    tensor = int_weight.reshape(-1, 32) - 8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    gptq_scale = gptq_scales.reshape(-1,1)
    # gptq_scale = torch.cat([gptq_scale,gptq_scale,gptq_scale,gptq_scale], dim=1).view(-1,1)
    pack_tensor = torch.cat((gptq_scale.half().view(torch.int8), tensor), dim=-1)
    pack_tensor.numpy().tofile(fout)
    print(f"converting {dst_name} qauntized tensor to ggml q4 block")

def convert_q4_1_tensor(src_name, dst_name, model, fout, q_config, n_head, n_head2=0, permute=False):
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    # g_idx = model[f"{src_name}.g_idx"]
    qweight = model[f"{src_name}.qweight"]
    int_weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros, q_config)

    int_weight = int_weight.view(-1,int_weight.shape[-1]).t()
    gptq_scales = gptq_scales.view(-1,gptq_scales.shape[-1]).t()
    gptq_zeros = gptq_zeros.view(-1,gptq_zeros.shape[-1]).t()

    write_header(fout, int_weight.shape, dst_name, 3)
    if permute:
        int_weight = permute_func(int_weight, n_head, n_head2).contiguous()
        gptq_scales = permute_func(gptq_scales, n_head, n_head2).contiguous()
        gptq_zeros = permute_func(gptq_zeros, n_head, n_head2).contiguous()

    tensor = int_weight.reshape(-1, 32)
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    gptq_scale = gptq_scales.reshape(-1,1)
    gptq_zeros = gptq_zeros.reshape(-1,1)
    gptq_zeros = -gptq_scale*gptq_zeros
    pack_tensor = torch.cat((gptq_scale.half().view(torch.int8), gptq_zeros.half().view(torch.int8), tensor), dim=-1)
    pack_tensor.numpy().tofile(fout)
    print(f"converting {dst_name} qauntized tensor to ggml q4 1 block")


def convert_q4_f32_tensor(src_name, dst_name, model, fout, q_config, n_head, n_head_kv=0, permute=False):
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    # g_idx = model[f"{src_name}.g_idx"]
    qweight = model[f"{src_name}.qweight"]

    int_weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros, q_config)
    # import pdb; pdb.set_trace()
    weight = (gptq_scales * (int_weight - gptq_zeros))
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    weight = weight.t()
    weight = weight.float()
    if permute:
        # weight = weight.t()
        weight = permute_func(weight, n_head, n_head_kv).contiguous()
        # weight = weight.t().contiguous()

    shape = weight.shape
    write_header(fout, shape, dst_name, 0)
    weight.numpy().tofile(fout)

    # import intel_extension_for_transformers.llm.runtime.graph.llama_cpp as cpp_model
    # dst = np.zeros((weight.shape[0], weight.shape[1]*4), dtype=np.int8)
    # byte_size = cpp_model.Model.np_jblas_quantize(weight.numpy(), dst,)
    #                                         #    weight_dtype="int4" if q_config['bits'] == 4 else "int8",
    #                                         #    group_size=q_config['group_size'],
    #                                         #    alg="sym",# if q_config['sym'] else "asym",
    #                                         #    compute_dtype="int8")
    # dst.flatten()[:byte_size].tofile(fout)
    # print(q_config)
    print(f"converting {dst_name} qauntized tensor to fp32 tensor")


def convert_q4_jblas_tensor(src_name, dst_name, model, fout, q_config, n_head, n_head_kv=0, permute=False):
    import intel_extension_for_transformers.llm.runtime.graph.llama_cpp as cpp_model
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    # g_idx = model[f"{src_name}.g_idx"]
    qweight = model[f"{src_name}.qweight"]
    int_weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros, q_config)
    
    int_weight = int_weight.view(-1,int_weight.shape[-1])
    gptq_scales = gptq_scales.view(-1,gptq_scales.shape[-1])
    gptq_zeros = gptq_zeros.view(-1,gptq_zeros.shape[-1])
    
    if permute:
        int_weight = permute_func(int_weight.t(), n_head, n_head_kv).t().contiguous()
        gptq_scales = permute_func(gptq_scales.t(), n_head, n_head_kv).t().contiguous()
        gptq_zeros = permute_func(gptq_zeros.t(), n_head, n_head_kv).t().contiguous()

    shape = int_weight.shape
    write_header(fout, shape[::-1], dst_name, 13)

    dst = np.zeros((int_weight.shape[0], int_weight.shape[1]*4), dtype=np.int8)
    # if q_config['sym']:
    # import pdb; pdb.set_trace()
    int_weight = int_weight - 8
    int_weight = int_weight * 16
    gptq_scales = gptq_scales / 16
    gptq_zeros = (gptq_zeros - 8) * 16
    import pdb; pdb.set_trace()
    byte_size = cpp_model.Model.np_jblas_qpack(int_weight.numpy(), gptq_scales.float().numpy(), gptq_zeros.numpy(), dst,
                                               weight_dtype="int4" if q_config['bits'] == 4 else "int8",
                                               group_size=q_config['group_size'],
                                               alg="sym" if q_config['sym'] else "asym",
                                               compute_dtype="fp32")
    dst.flatten()[:byte_size].tofile(fout)
    print(f"converting {dst_name} qauntized tensor to jblas q4 block")



def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    out_path = args.outfile.as_posix()
    model_path = args.model.as_posix()

    model, quantize_config = load_gptq_model(model_path)
    f = open(out_path, "wb")
    
    # 1. write hparams
    n_vocab, n_embd = model['model.embed_tokens.weight'].shape
    layer_re = r'model\.layers\.([0-9]+)'
    n_layer = 1 + max(int(re.match(layer_re, name).group(1)) for name in model
                        if re.match(layer_re, name))

    # hardcoded:
    n_mult = 256
    n_head = {32: 32, 40: 40, 60: 52, 80: 64}[n_layer]
    ffn_hidden_size = 14336
    # 1. write head and params
    f.write(b"ggjt"[::-1])  # magic

    n_head = n_head
    n_head_kv = 8
    values = [
        1,  # file version
        n_vocab,
        n_embd,
        256, #hparams.n_mult,
        n_head,
        n_head_kv, # n_head_kv (multi_query attention)
        n_layer,
        n_embd // n_head,  # rot (obsolete)
        0, #file_type.value, # TODO
    ]
    # import pdb; pdb.set_trace()
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

    f.write(struct.pack("i", 1)) # TODO, bos_token_id = 0 in https://huggingface.co/decapoda-research/llama-7b-hf/blob/main/config.json but bos_token_id = 1 in llama.cpp
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

    for i in range(n_layer):
        convert_q4_1_tensor(f"model.layers.{i}.self_attn.q_proj",
                    f"layers.{i}.attention.wq.weight", list_vars, f, quantize_config, n_head, n_head, permute=True)
        convert_q4_1_tensor(f"model.layers.{i}.self_attn.k_proj",
                    f"layers.{i}.attention.wk.weight", list_vars, f, quantize_config, n_head, n_head_kv, permute=True)
        convert_q4_1_tensor(f"model.layers.{i}.self_attn.v_proj",
                    f"layers.{i}.attention.wv.weight", list_vars, f, quantize_config, n_head)
        convert_q4_1_tensor(f"model.layers.{i}.self_attn.o_proj",
                    f"layers.{i}.attention.wo.weight", list_vars, f, quantize_config, n_head)
        convert_q4_1_tensor(f"model.layers.{i}.mlp.gate_proj",
                    f"layers.{i}.feed_forward.w1.weight", list_vars, f, quantize_config, n_head)
        convert_q4_1_tensor(f"model.layers.{i}.mlp.down_proj",
                    f"layers.{i}.feed_forward.w2.weight", list_vars, f, quantize_config, n_head)
        convert_q4_1_tensor(f"model.layers.{i}.mlp.up_proj",
                    f"layers.{i}.feed_forward.w3.weight", list_vars, f, quantize_config, n_head)

        convert_fp32_tensor(f"model.layers.{i}.input_layernorm.weight",
                        f"layers.{i}.attention_norm.weight", list_vars, f)
        convert_fp32_tensor(f"model.layers.{i}.post_attention_layernorm.weight",
                        f"layers.{i}.ffn_norm.weight", list_vars, f)


    f.close()
    print(f"Success! saved as {out_path}")

if __name__ == '__main__':
    main()