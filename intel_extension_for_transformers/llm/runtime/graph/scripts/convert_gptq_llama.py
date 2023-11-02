import torch
import os
import numpy as np
import struct
from transformers import AutoTokenizer, TextStreamer
from transformers import AutoModelForCausalLM
import json
import sys
import copy
import re
from neural_compressor.adaptor.torch_utils.weight_only import quant_weight, quant_weight_w_scale
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Literal, Optional, Sequence, Tuple, TypeVar, Union)
from pathlib import Path
from sentencepiece import SentencePieceProcessor  # type: ignore

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
            raise Exception(f"Expected added token IDs to be sequential and start at {len(added_tokens)}; got {actual_ids}")
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
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

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
            raise FileNotFoundError(f"Could not find tokenizer.model in {path} or its parent; if it's in another directory, pass the directory as --vocab-dir")
    added_tokens_path = path.parent / "added_tokens.json"
    print(f"Loading vocab file {path}")
    return SentencePieceVocab(path, added_tokens_path if added_tokens_path.exists() else None)


def permute_func(weights, n_head: int, n_head_kv: int):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head //= n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

def recover_weight(qweight, scales, qzeros, permute=False, group_size=128, bits=4):
    wf = torch.tensor([[ 0,  4,  8, 12, 16, 20, 24, 28]], dtype=torch.int32)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(zeros, (2 ** bits) - 1, out=zeros)
        
    zeros = zeros + 1
    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    scales = scales
    scales = scales.reshape(-1, 1, scales.shape[-1])
        
    weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight,(2 ** bits) - 1, out=weight)
    weight = weight.reshape(-1, group_size, weight.shape[2])

    if permute:
        out1 = permute_func(weight.view(-1,weight.shape[-1]).t().numpy(), 32, 32)
    else:
        out1 = weight.view(-1,weight.shape[-1]).t().numpy()
    tensor = torch.tensor(out1).reshape(-1, 32) #+ 8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)

    if permute:
        out2 = permute_func(scales.view(-1,scales.shape[-1]).t().numpy(), 32, 32)
    else:
        out2 = scales.view(-1,scales.shape[-1]).t().numpy()
    gptq_scale = torch.tensor(out2).reshape(-1,1)
    gptq_scale = torch.cat([gptq_scale,gptq_scale,gptq_scale,gptq_scale], dim=1).view(-1,1)
    pack_tensor = torch.cat((gptq_scale.half().view(torch.int8), tensor), dim=-1)

    weight = (scales * (weight - zeros))
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    return weight.t(), pack_tensor

# 3. write tensors
def write_header(fout, shape, dst_name, ftype_cur):
    sname = dst_name.encode('utf-8')
    fout.write(struct.pack("iii", len(shape), len(sname), ftype_cur))
    fout.write(struct.pack("i" * len(shape), *shape[::-1]))
    fout.write(sname)
    fout.seek((fout.tell() + 31) & -32)

def convert_non_q4(src_name, dst_name, model, fout):
    v = model[src_name]
    shape = v.shape
    print("Processing non-Q4 variable: " + src_name +
          " with shape: ", shape, " and type: ", v.dtype)
    # if len(shape) == 1:
    print("  Converting to float32")
    v = v.to(torch.float32)

    ftype_cur = {torch.float16: 1, torch.float32: 0}[v.dtype]

    # header
    write_header(fout, shape, dst_name, ftype_cur)

    # data
    v.numpy().tofile(fout)

def expandToInt4(qweight):
    eweight = qweight.repeat(8, axis=2)
    eweight = eweight.astype(np.uint32)
    for i in range(0, eweight.shape[2]):
        offset = i % (32 // 4) * 4
        eweight[:, :, i] = eweight[:, :, i] >> offset & (2 ** 4 - 1)
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
            zeros[:, j] = (qzeros[:, col] >> (bits * (j - i)) & (2 ** bits - 1)) + 1
        i += 32 // bits
        col += 1
    return zeros

def convert_q4_recover(src_name, dst_name, model, fout, n_head, n_head2=0, permute=False):
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    g_idx = model[f"{src_name}.g_idx"]
    qweight = model[f"{src_name}.qweight"]  # transpose

    weight,_ = recover_weight(qweight, scales, qzeros)
    
    src_name = src_name + ".weight"
    shape = weight.shape
    weight = weight.to(torch.float32)
    write_header(fout, shape, dst_name, 0)
    if permute:
        tensor = permute_func(weight.numpy(), n_head, n_head2)
        tensor.tofile(fout)
    else:
        # tensor = quantize_q4_0(v)
        weight.numpy().tofile(fout)
    print("converting q4 0 recover")

def convert_q4(src_name, dst_name, model, fout, n_head, n_head2=0, permute=False):
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    g_idx = model[f"{src_name}.g_idx"]
    qweight = model[f"{src_name}.qweight"]  # transpose

    weight, pack = recover_weight(qweight, scales, qzeros, permute)
    
    src_name = src_name + ".weight"
    shape = weight.shape
    weight = weight.to(torch.float32)
    write_header(fout, shape, dst_name, 2)
    pack.numpy().tofile(fout)
    print("converting q4 0")


def main(model_path, out_path):
    # model_name = "/mnt/disk1/data2/zhenweil/models/gptq/Llama-2-7B-Chat-GPTQ"
    from safetensors.torch import load_file
    model = load_file(model_path + "/model.safetensors")

    f = open(out_path, "wb")
    
    # 1. write hparams
    n_vocab, n_embd = model['model.embed_tokens.weight'].shape
    layer_re = r'model\.layers\.([0-9]+)'
    n_layer = 1 + max(int(re.match(layer_re, name).group(1)) for name in model
                        if re.match(layer_re, name))

    # hardcoded:
    n_mult = 256
    n_head = {32: 32, 40: 40, 60: 52, 80: 64}[n_layer]

    # 1. write head and params
    f.write(b"ggjt"[::-1])  # magic

    n_head = n_head
    n_head_kv = n_head
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
    f.write(struct.pack("i" * len(values), *values))
    f.write(struct.pack("i", 0))
    f.write(struct.pack("f", 0))
    f.write(struct.pack("f", 0))
    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    f.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    f.write(struct.pack("i", 0))
    f.write(struct.pack("i", 11008))
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
    convert_non_q4("model.embed_tokens.weight", "tok_embeddings.weight", list_vars, f)
    convert_non_q4("model.norm.weight", "norm.weight", list_vars, f)
    convert_non_q4("lm_head.weight", "output.weight", list_vars, f)

    for i in range(n_layer):
        convert_q4(f"model.layers.{i}.self_attn.q_proj",
                    f"layers.{i}.attention.wq.weight", list_vars, f, n_head, n_head, permute=True)
        convert_q4(f"model.layers.{i}.self_attn.k_proj",
                    f"layers.{i}.attention.wk.weight", list_vars, f, n_head, n_head_kv, permute=True)
        convert_q4(f"model.layers.{i}.self_attn.v_proj",
                    f"layers.{i}.attention.wv.weight", list_vars, f, n_head)
        convert_q4(f"model.layers.{i}.self_attn.o_proj",
                    f"layers.{i}.attention.wo.weight", list_vars, f, n_head)
        convert_q4(f"model.layers.{i}.mlp.gate_proj",
                    f"layers.{i}.feed_forward.w1.weight", list_vars, f, n_head)
        convert_q4(f"model.layers.{i}.mlp.down_proj",
                    f"layers.{i}.feed_forward.w2.weight", list_vars, f, n_head)
        convert_q4(f"model.layers.{i}.mlp.up_proj",
                    f"layers.{i}.feed_forward.w3.weight", list_vars, f, n_head)

        convert_non_q4(f"model.layers.{i}.input_layernorm.weight",
                        f"layers.{i}.attention_norm.weight", list_vars, f)
        convert_non_q4(f"model.layers.{i}.post_attention_layernorm.weight",
                        f"layers.{i}.ffn_norm.weight", list_vars, f)


    f.close()
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: convert-gptq-to-ggml.py gptq_model_path out.bin\n")
        sys.exit(1)

    model_path = sys.argv[1]
    out_path = sys.argv[2]
    main(model_path, out_path)
