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
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypeVar,
                    Union)
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
