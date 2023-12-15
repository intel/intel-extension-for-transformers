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
from pathlib import Path
import numpy as np
import struct
import json
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Literal, Optional, Sequence, Tuple, TypeVar, Union)
from sentencepiece import SentencePieceProcessor  # type: ignore

GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32
GGML_QK5_0 = 32
GGML_QK5_1 = 32

GGML_QK4_0_TYPE = 2
GGML_QK4_1_TYPE = 3
GGML_QJBLAS_TYPE = 13

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

def quantize_q4_1(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_1 in ggml.c
    assert tensor.shape[1] % GGML_QK4_1 == 0
    tensor = tensor.view(-1, GGML_QK4_1)
    min_vals = tensor.min(dim=-1, keepdim=True).values
    max_vals = tensor.max(dim=-1, keepdim=True).values
    scale = (max_vals - min_vals) / ((1 << 4) - 1)
    tensor = ((tensor - min_vals) / scale).round().clamp(min=0, max=15).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale & min into each block
    import pdb; pdb.set_trace()
    tensor = torch.cat((scale.half().view(torch.int8), min_vals.half().view(torch.int8), tensor), dim=-1)
    return tensor

class SentencePieceVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path]) -> None:
        self.sentencepiece_tokenizer = SentencePieceProcessor(str(fname_tokenizer))
        added_tokens: Dict[str, int]
        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens))
        else:
            added_tokens = {}
        vocab_size: int = self.sentencepiece_tokenizer.vocab_size()
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            raise Exception(
                f"Expected added token IDs to be sequential and start at {len(added_tokens)}; got {actual_ids}")
        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_list = [text for (text, idx) in items]
        self.vocab_size_base: int = vocab_size
        self.vocab_size: int = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def sentencepiece_tokens(self) -> Iterable[Tuple[bytes, float]]:
        tokenizer = self.sentencepiece_tokenizer
        for i in range(tokenizer.vocab_size()):
            text: bytes
            if tokenizer.is_unknown(i):
                text = " \u2047 ".encode("utf-8")
            elif tokenizer.is_control(i):
                text = b""
            elif tokenizer.is_byte(i):
                piece = tokenizer.id_to_piece(i)
                if len(piece) != 6:
                    raise Exception(f"Invalid token: {piece}")
                byte_value = int(piece[3:-1], 16)
                text = struct.pack("B", byte_value)
            else:
                text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
            score: float = tokenizer.get_score(i)
            yield text, score

    def added_tokens(self) -> Iterable[Tuple[bytes, float]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)}\
                added tokens>"


def load_vocab(path: Path) -> SentencePieceVocab:
    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    if path.is_dir():
        path2 = path / "tokenizer.model"
        # Use `.parent` instead of /.. to handle the symlink case better.
        path3 = path.parent / "tokenizer.model"
        if path2.exists():
            path = path2
        elif path3.exists():
            path = path3
        else:
            raise FileNotFoundError(
                f"Could not find tokenizer.model in {path} or its parent; if it's in another directory, \
                pass the directory as --vocab-dir"
            )
    added_tokens_path = path.parent / "added_tokens.json"
    print(f"Loading vocab file {path}")
    return SentencePieceVocab(path, added_tokens_path if added_tokens_path.exists() else None)


def expandToInt4(qweight):
    eweight = qweight.repeat(8, axis=2)
    eweight = eweight.astype(np.uint32)
    for i in range(0, eweight.shape[2]):
        offset = i % (32 // 4) * 4
        eweight[:, :, i] = eweight[:, :, i] >> offset & (2**4 - 1)
    return eweight


def to_ggml_int16(eweight):
    qweight = np.zeros((eweight.shape[0], eweight.shape[1], eweight.shape[2] // 4), dtype=np.uint16)
    eweight = np.asarray(eweight, dtype=np.uint16)
    for i in range(0, qweight.shape[2]):
        qweight[:, :, i] = eweight[:, :, i * 2 + 0]
        qweight[:, :, i] |= eweight[:, :, i * 2 + 32] << 1 * 4
        qweight[:, :, i] |= eweight[:, :, i * 2 + 1] << 2 * 4
        qweight[:, :, i] |= eweight[:, :, i * 2 + 33] << 3 * 4
    return qweight.astype(np.int16)


def qzeros_to_zeros(qzeros, bits=4):
    zeros = np.zeros((qzeros.shape[0], qzeros.shape[1] * (32 // bits)), dtype=np.float32)
    i = 0
    col = 0
    while col < qzeros.shape[1]:
        for j in range(i, i + (32 // bits)):
            zeros[:, j] = (qzeros[:, col] >> (bits * (j - i)) & (2**bits - 1)) + 1
        i += 32 // bits
        col += 1
    return zeros


def unpack_weight(qweight, scales, qzeros, q_config):
    group_size = q_config['group_size']
    bits = q_config['bits']
    wf = torch.tensor([[ 0,  4,  8, 12, 16, 20, 24, 28]], dtype=torch.int32)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
                                      wf.unsqueeze(0)).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(zeros, (2 ** bits) - 1, out=zeros)
        
    zeros = zeros + 1
    # zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
    zeros = zeros.reshape(scales.shape)

    # scales = scales
    # scales = scales.reshape(-1, 1, scales.shape[-1])
        
    weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
                                       wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight,(2 ** bits) - 1, out=weight)
    # int_weight = weight.reshape(-1, group_size, weight.shape[2])

    return weight, scales, zeros

def write_header(fout, shape, dst_name, ftype_cur):
    sname = dst_name.encode('utf-8')
    fout.write(struct.pack("iii", len(shape), len(sname), ftype_cur))
    fout.write(struct.pack("i" * len(shape), *shape[::-1]))
    fout.write(sname)
    fout.seek((fout.tell() + 31) & -32)


def find_quantized_model_file(model_path):
    model_path = Path(model_path)
    for ext in ['.safetensors', '.pt']:
        found = list(model_path.glob(f"*{ext}"))
        if len(found) > 0:
            if len(found) != 1:
                warnings.warn(f'Detected {len(found)} {ext} model, use the first one {found[0]}.')
            print(f"Detected model file {found[0]}")
            return str(found[0])

def load_gptq_model(model_path):
    input_path = find_quantized_model_file(model_path)
    model = None
    if input_path.endswith('pt'):
        model = torch.load(input_path, map_location="cpu")
    elif input_path.endswith('safetensors'):
        from safetensors.torch import load_file
        model = load_file(input_path)
    else:
        print("unknown input model path, only support .safetensors or .pt file.")

    with open(model_path + '/config.json', "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(model_path + '/quantize_config.json', "r", encoding="utf-8") as f:
        quantize_config = json.load(f)
    return model, config, quantize_config


def convert_fp32_tensor(src_name, dst_name, model, fout):
    v = model[src_name]
    shape = v.shape
    # print("Processing non-Q4 variable: " + src_name +
    #       " with shape: ", shape, " and type: ", v.dtype)
    v = v.to(torch.float32)

    ftype_cur = {torch.float16: 1, torch.float32: 0}[v.dtype]

    # header
    write_header(fout, shape, dst_name, ftype_cur)

    # data
    v.numpy().tofile(fout)
    print(f"converting {dst_name} float tensor")

def convert_q4_tensor(src_name, dst_name, model, fout, q_config, n_head, n_head2=0, permute_func=None):
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    qweight = model[f"{src_name}.qweight"]
    int_weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros, q_config)

    int_weight = int_weight.view(-1,int_weight.shape[-1]).t()
    gptq_scales = gptq_scales.view(-1,gptq_scales.shape[-1]).t()
    gptq_zeros = gptq_zeros.view(-1,gptq_zeros.shape[-1]).t()

    write_header(fout, int_weight.shape, dst_name, 2)
    if permute_func:
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

def convert_q4_1_tensor(src_name, dst_name, model, fout, q_config, n_head, n_head2=0, permute_func=None):
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
    if permute_func:
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


def convert_q4_f32_tensor(src_name, dst_name, model, fout, q_config, n_head, n_head_kv=0, permute_func=None):
    qzeros = model[f"{src_name}.qzeros"]
    scales = model[f"{src_name}.scales"]
    qweight = model[f"{src_name}.qweight"]

    weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros, q_config)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    # num_itr = g_idx.shape[0]//x.shape[-1]
    if q_config['desc_act']:
        g_idx = model[f"{src_name}.g_idx"]
        weight = (gptq_scales[g_idx.long()] * (weight - gptq_zeros[g_idx.long()]))
    else:
        infeatures = weight.shape[0]
        g_idx = torch.tensor([i // q_config["group_size"] for i in range(infeatures)], dtype=torch.int32)
        weight = (gptq_scales[g_idx.long()] * (weight - gptq_zeros[g_idx.long()]))
    
    weight = weight.t()
    weight = weight.float()
    if permute_func:
        weight = permute_func(weight, n_head, n_head_kv).contiguous()

    shape = weight.shape
    write_header(fout, shape, dst_name, 0)
    weight.numpy().tofile(fout)

    print(f"converting {dst_name} qauntized tensor to fp32 tensor")


def convert_q4_jblas_tensor(src_name, dst_name, model, fout, q_config, n_head, n_head_kv=0, permute_func=None):
    import intel_extension_for_transformers.llm.runtime.graph.llama_cpp as cpp_model
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    qweight = model[f"{src_name}.qweight"]

    int_weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros, q_config)
    int_weight = int_weight.view(-1,int_weight.shape[-1])

    if permute_func:
        int_weight = permute_func(int_weight.t(), n_head, n_head_kv).t().contiguous()
        gptq_scales = permute_func(gptq_scales.t(), n_head, n_head_kv).t().contiguous()
        gptq_zeros = permute_func(gptq_zeros.t(), n_head, n_head_kv).t().contiguous()

    if q_config['desc_act']:
        g_idx = model[f"{src_name}.g_idx"]
        int_weight2 = int_weight.clone()
        group_size=q_config['group_size']
        group_dict = {}
        for i in range(len(g_idx)):
            group_idx = g_idx[i].item()
            if group_idx not in group_dict:
                target_idx = group_idx * group_size
                group_dict[group_idx] = 0
            else:
                group_dict[group_idx] = group_dict[group_idx] + 1
                target_idx = group_idx * group_size + group_dict[group_idx]
            int_weight2[target_idx] = int_weight[i]
        int_weight = int_weight2

    shape = int_weight.shape
    write_header(fout, shape[::-1], dst_name, GGML_QJBLAS_TYPE)

    dst = np.zeros((int_weight.shape[0], int_weight.shape[1]*4), dtype=np.int8)
    int_weight = np.ascontiguousarray(((int_weight - 8) * 16).numpy())
    gptq_scales = np.ascontiguousarray((gptq_scales.float() / 16).numpy())
    if q_config['sym']:
        gptq_zeros = np.empty(0, dtype=np.int8)
    else:
        gptq_zeros = np.ascontiguousarray(((gptq_zeros - 8) * 16).numpy())
    if q_config['desc_act']:
        g_idx = np.ascontiguousarray(g_idx.numpy())
    else:
        g_idx = np.empty(0, dtype=np.int32)

    byte_size = cpp_model.Model.np_jblas_qpack(int_weight, gptq_scales, gptq_zeros, g_idx, dst,
                                               weight_dtype="int4" if q_config['bits'] == 4 else "int8",
                                               group_size=q_config['group_size'],
                                               alg="sym" if q_config['sym'] else "asym",
                                               compute_dtype="int8")
    dst.flatten()[:byte_size].tofile(fout)
    print(f"converting {dst_name} qauntized tensor to jblas q4 block")