"""
Convert Hugging Face ChatGLM/ChatGLM2 models to GGML format
"""
import argparse
import platform
import struct
import sys
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Optional

import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32
GGML_QK5_0 = 32
GGML_QK5_1 = 32

GGML_MEM_ALIGN = 16

if platform.system() == "Darwin":
    # cpm_kernels doesn't support macOS but transformers will check missing packages, so mock it
    sys.modules["cpm_kernels"] = object()


class GGMLType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q4_J = 13


class ModelType(Enum):
    CHATGLM = 1
    CHATGLM2 = 2
    BAICHUAN7B = 1024
    BAICHUAN13B = 1025


def quantize_q8_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q8_0 in ggml.c
    assert tensor.shape[1] % GGML_QK8_0 == 0
    tensor = tensor.view(-1, GGML_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).char()
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


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

def quantize_q4_j(tensor: torch.Tensor, f):
    import intel_extension_for_transformers.llm.runtime.graph.chatglm2_cpp as cpp_model
    import numpy as np
    dst = np.zeros((tensor.shape[0], tensor.shape[1]*2), dtype=np.int8)
    byte_size = cpp_model.Model.np_jblas_quantize(tensor.numpy(), dst)
    # print("tensor shape: ", tensor.shape)
    # print("byte_size: ", byte_size)
    import struct
    dst = dst.flatten()
    f.write(struct.pack('b' * byte_size, *list(dst[:byte_size])))
    # for i in range(byte_size):
    #     f.write(struct.pack('b', dst[i]))



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
    tensor = torch.cat((scale.half().view(torch.int8), min_vals.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q5_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q5_0 in ggml.c
    assert tensor.shape[1] % GGML_QK5_0 == 0
    tensor = tensor.view(-1, GGML_QK5_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -16
    tensor = (tensor / scale + 16).round().clamp(min=0, max=31).char()
    qs = (tensor[:, :16] & 0x0F) | (tensor[:, 16:] << 4)
    qh = torch.zeros(tensor.shape[:-1], dtype=torch.int32)
    for i in range(32):
        qh |= ((tensor[:, i] & 0x10) >> 4).int() << i

    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), qh[..., None].view(torch.int8), qs), dim=-1)
    return tensor


def quantize_q5_1(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q5_1 in ggml.c
    assert tensor.shape[1] % GGML_QK5_1 == 0
    tensor = tensor.view(-1, GGML_QK5_1)
    min_vals = tensor.min(dim=-1, keepdim=True).values
    max_vals = tensor.max(dim=-1, keepdim=True).values
    scale = (max_vals - min_vals) / ((1 << 5) - 1)
    tensor = ((tensor - min_vals) / scale).round().clamp(min=0, max=31).char()
    qs = (tensor[:, :16] & 0x0F) | (tensor[:, 16:] << 4)
    qh = torch.zeros(tensor.shape[:-1], dtype=torch.int32)
    for i in range(32):
        qh |= ((tensor[:, i] & 0x10) >> 4).int() << i

    # add scale & min into each block
    tensor = torch.cat(
        (scale.half().view(torch.int8), min_vals.half().view(torch.int8), qh[..., None].view(torch.int8), qs), dim=-1
    )
    return tensor


def dump_tensor(f, name: str, tensor: torch.Tensor, ggml_type: GGMLType):
    assert tensor.dtype == torch.float32

    # tensor name
    # f.write(struct.pack("i", len(name.encode())))
    # f.write(name.encode())

    # # tensor shape & dtype
    # f.write(struct.pack("i" * (2 + tensor.ndim), tensor.ndim, *tensor.shape, ggml_type.value))
    data = tensor.squeeze()
    n_dims = len(data.shape)
    str = name.encode("utf-8")
    if name == 'transformer.embedding.word_embeddings.weight':
        f.write(struct.pack("iii", n_dims, len(str), 2))
    else:
        f.write(struct.pack("iii", n_dims, len(str), ggml_type.value))
    for i in range(n_dims):
        f.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    f.write(str)
    # TODO: file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
    # fout.seek((fout.tell() + 31) & -32)
    # tensor data

    if ggml_type != GGMLType.Q4_J:
        if ggml_type == GGMLType.F32:
            tensor = tensor.float()
        elif ggml_type == GGMLType.F16:
            tensor = tensor.half()
        elif ggml_type == GGMLType.Q8_0:
            tensor = quantize_q8_0(tensor)
        elif ggml_type == GGMLType.Q4_0:
            tensor = quantize_q4_0(tensor)
        elif ggml_type == GGMLType.Q4_1:
            tensor = quantize_q4_1(tensor)
        elif ggml_type == GGMLType.Q5_0:
            tensor = quantize_q5_0(tensor)
        elif ggml_type == GGMLType.Q5_1:
            tensor = quantize_q5_1(tensor)
        else:
            raise NotImplementedError(f"Cannot dump tensor of dtype {tensor.dtype}")

        # align address
        # aligned_pos = (f.tell() + (GGML_MEM_ALIGN - 1)) // GGML_MEM_ALIGN * GGML_MEM_ALIGN
        # f.seek(aligned_pos)
        tensor.numpy().tofile(f)
    else:
        if name == 'transformer.embedding.word_embeddings.weight':
            tensor = quantize_q4_0(tensor)
            tensor.numpy().tofile(f)
        else:
            # import pdb; pdb.set_trace()
            quantize_q4_j(tensor, f)


def dump_state_dict(f, weight_names, state_dict, quantization_bit, ggml_type):
    tensor_info = []
    for name in tqdm(weight_names, desc="Processing model states"):
        tensor = state_dict[name]
        if tensor.ndim == 2:
            # 2d weight: should quantize it if needed

            # step 1: de-quantize it back to float32
            if tensor.dtype == torch.int8:
                assert quantization_bit in [4, 8]
                scale = state_dict[f"{name}_scale"].float()  # channel-wise scale

                if quantization_bit == 4:
                    # convert int4 weight to int8
                    low_bits = ((tensor << 4) & 0xF0) >> 4
                    high_bits = (tensor & 0xF0) >> 4
                    tensor = torch.stack((high_bits, low_bits), dim=-1).view(tensor.shape[0], -1)
                tensor = tensor * scale[:, None]
            else:
                tensor = tensor.float()

            # step 2: quantize it into ggml format
            tensor_ggml_type = ggml_type
        else:
            # 1d weight: convert it to float32
            assert tensor.ndim == 1
            tensor = tensor.float()
            tensor_ggml_type = GGMLType.F32
        # import pdb; pdb.set_trace()

        dump_tensor(f, name, tensor, tensor_ggml_type)
        tensor_info.append((name, tensor.shape, tensor_ggml_type.name))

    print(tabulate(tensor_info, headers=["name", "shape", "dtype"], tablefmt="psql"))


class BaseConverter:
    @classmethod
    def convert(cls, f, model, tokenizer, ggml_type):
        # f.write(b"ggml")  # magic
        # f.write(struct.pack("ii", cls.MODEL_TYPE.value, 1))  # model type & version
        f.write(struct.pack("i", 0x67676d66))
        f.write(struct.pack("i", 1))
        cls.dump_config(f, model.config, ggml_type)
        cls.dump_tokenizer(f, tokenizer)
        cls.dump_model(f, model, ggml_type)
import sentencepiece.sentencepiece_model_pb2 as model
from sentencepiece import SentencePieceProcessor  # type: ignore
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Literal, Optional, Sequence, Tuple, TypeVar, Union)
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

class ChatGLM2Converter(BaseConverter):
    MODEL_TYPE = ModelType.CHATGLM2

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.add_bias_linear is False, "unimplemented: add_bias_linear must be false"
        assert config.add_qkv_bias is True, "unimplemented: add_qkv_bias must be true"
        assert (
            config.apply_residual_connection_post_layernorm is False
        ), "unimplemented: apply_residual_connection_post_layernorm must be false"
        assert (
            config.kv_channels * config.num_attention_heads == config.hidden_size
        ), "unimplemented: invalid kv_channels"
        assert config.multi_query_attention is True, "unimplemented: multi_query_attention must be true"
        assert config.original_rope is True, "unimplemented: original_rope must be true"
        assert config.post_layer_norm is True, "unimplemented: post_layer_norm must be true"
        assert config.rmsnorm is True, "unimplemented: rmsnorm must be true"

        config_values = [
            # ggml_type.value,
            10,
            config.padded_vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_layers,
            config.ffn_hidden_size,
            config.seq_length,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.multi_query_group_num,
        ]

        # f.write(struct.pack("i" * len(config_values), *config_values))
        hparams = config.to_dict()
        f.write(struct.pack("i", hparams["padded_vocab_size"]))
        f.write(struct.pack("i", hparams["hidden_size"]))
        f.write(struct.pack("i", 10))
        f.write(struct.pack("i", hparams["num_attention_heads"]))
        f.write(struct.pack("i", 0))
        f.write(struct.pack("i", hparams["num_layers"]))
        f.write(struct.pack("i", 0))
        f.write(struct.pack("i", 0))
        f.write(struct.pack("i", hparams["seq_length"]))
        f.write(struct.pack("f", 0))
        f.write(struct.pack("f", 0))
        f.write(struct.pack("i", 0))

        f.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
        f.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

        f.write(struct.pack("i", hparams["multi_query_group_num"]))
        f.write(struct.pack("i", hparams["ffn_hidden_size"]))
        f.write(struct.pack("i", 0))

    @staticmethod
    def dump_tokenizer(f, tokenizer):
        f.write(struct.pack("i", tokenizer.bos_token_id if tokenizer.bos_token_id else 1))
        f.write(struct.pack("i", tokenizer.eos_token_id if tokenizer.eos_token_id else 2))
        f.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id else -1))
        f.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id else -1))


        vocab = load_vocab(Path("/mnt/disk1/data2/zhenweil/models/chatglm2-6b"))
        counter = 0
        for text, score in vocab.all_tokens():
            f.write(struct.pack("i", len(text)))
            f.write(text)
            f.write(struct.pack("f", score))
            counter += 1

        while counter < 65024:
            f.write(struct.pack("i", len(text)))
            f.write(text)
            f.write(struct.pack("f", 0))
            counter += 1
        # counter = 0
        # import pdb; pdb.set_trace()
        # for text, score in tokenizer.all_tokens():
        #     f.write(struct.pack("i", len(text)))
        #     f.write(text)
        #     f.write(struct.pack("f", score))
        #     counter += 1

        # while counter < hparams["padded_vocab_size"]:
        #     f.write(struct.pack("i", len(text)))
        #     f.write(text)
        #     f.write(struct.pack("f", 0))
        #     counter += 1

    @staticmethod
    def dump_model(f, model, ggml_type):
        weight_names = ["transformer.embedding.word_embeddings.weight"]
        for i in range(model.config.num_layers):
            weight_names += [
                f"transformer.encoder.layers.{i}.input_layernorm.weight",
                f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight",
                f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias",
                f"transformer.encoder.layers.{i}.self_attention.dense.weight",
                f"transformer.encoder.layers.{i}.post_attention_layernorm.weight",
                f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight",
                f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight",
            ]
        weight_names += [
            "transformer.encoder.final_layernorm.weight",
            "transformer.output_layer.weight",
        ]
        dump_state_dict(f, weight_names, model.state_dict(), model.config.quantization_bit, ggml_type)


def convert(f: BinaryIO, model_name_or_path: str, lora_model_name_or_path: Optional[str] = None, dtype: str = "q4_0"):
    ggml_type = GGMLType[dtype.upper()]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if "AutoModel" in config.auto_map:
        auto_model_class = AutoModel
    elif "AutoModelForCausalLM" in config.auto_map:
        auto_model_class = AutoModelForCausalLM
    else:
        raise RuntimeError(f"Cannot find auto model class to load {model_name_or_path}")

    model = auto_model_class.from_pretrained(model_name_or_path, trust_remote_code=True, low_cpu_mem_usage=True)

    if lora_model_name_or_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_model_name_or_path)
        model = model.merge_and_unload()

    if model.config.model_type == "chatglm":
        if hasattr(model.config, "multi_query_attention"):
            ChatGLM2Converter.convert(f, model, tokenizer, ggml_type)
        else:
            ChatGLMConverter.convert(f, model, tokenizer, ggml_type)
    elif model.config.model_type == "baichuan":
        if model.config.hidden_size == 5120:
            Baichuan13BConverter.convert(f, model, tokenizer, ggml_type)
        else:
            Baichuan7BConverter.convert(f, model, tokenizer, ggml_type)
    else:
        raise RuntimeError(f"Unknown model type {model.config.model_type}")


def main():
    parser = argparse.ArgumentParser("chatglm-convert")
    parser.add_argument(
        "-i",
        "--model_name_or_path",
        default="THUDM/chatglm-6b",
        type=str,
        help="Model name or path used in AutoModel.from_pretrained",
    )
    parser.add_argument(
        "-l",
        "--lora_model_name_or_path",
        default=None,
        type=str,
        help="Lora model name or path used in PeftModel.from_pretrained",
    )
    parser.add_argument(
        "-o", "--save_path", default="chatglm-ggml.bin", type=Path, help="Path to save the generated GGML model"
    )
    parser.add_argument(
        "-t",
        "--type",
        default="q4_0",
        type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q4_j"],
        help="GGML model quantization type",
    )
    args = parser.parse_args()

    with open(args.save_path, "wb") as f:
        convert(f, args.model_name_or_path, dtype=args.type)

    print(f"GGML model saved to {args.save_path}")


if __name__ == "__main__":
    main()