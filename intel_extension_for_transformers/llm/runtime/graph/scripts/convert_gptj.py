#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# Convert Hugging Face fine-tuned gpt-neox-like models to ne format
#
# Usage:
#
#   python3 models/convert-h5-to-ne.py
#
# This script is similar to "convert-pt-to-ne.py"
#

import sys
import struct
import json
import torch
import numpy as np
from pathlib import Path
import argparse
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypeVar,
                    Union)
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    dir_model = args.model.as_posix()
    fname_out = args.outfile.as_posix()

    ftype = 0
    if args.outtype == "f16":
        ftype = 1

    print("Loading model: ", dir_model)
    tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True)
    hparams = model.config.to_dict()
    list_vars = model.state_dict()
    fout = open(fname_out, "wb")

    fout.write(b"ggjt"[::-1])  #0x67676d6c)) # magic: ggml in hex
    values = [
        1,  # file version
        hparams["vocab_size"],
        hparams["n_embd"],
        hparams["n_embd"] // hparams["n_head"],
        hparams["n_head"],
        hparams.get("n_head_kv", 0),  # multi-query attention
        hparams["n_layer"],
        hparams["rotary_dim"],
        ftype
    ]
    fout.write(struct.pack("i" * len(values), *values))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("f", 0))
    fout.write(struct.pack("f", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    fout.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("f", hparams.get("rms_norm_eps", 1e-6)))  # rms norm eps
    fout.write(struct.pack("f", 10000.0))  # freq_base
    
    fout.write(struct.pack("i", tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1))
    fout.write(struct.pack("i", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2))
    fout.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    encoder = tokenizer.vocab
    # Add added_tokens (special tokens) to the encoder
    encoder_added = tokenizer.get_added_vocab()

    for i, key in enumerate(sorted(encoder, key=encoder.get)):
        # for key in encoder:
        text = bytearray([byte_decoder[c] for c in key])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        if key not in encoder_added:
            fout.write(struct.pack("f", 0.0 - i))
        else:
            fout.write(struct.pack("f", -10000))

    for name in list_vars.keys():
        data = list_vars[name].squeeze().numpy()
        print("Processing variable: " + name + " with shape: ", data.shape)

        # we don't need these
        if name.endswith("attn.masked_bias") or name.endswith(".attn.bias"):
            print("  Skipping variable: " + name)
            continue

        n_dims = len(data.shape)

        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype_cur = 0
        if ftype != 0:
            if name[-7:] == ".weight" and n_dims == 2:
                print("  Converting to float16")
                data = data.astype(np.float16)
                ftype_cur = 1
            else:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0
        else:
            if data.dtype != np.float32:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0

        str = name.encode('utf-8')
        shape = data.shape
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        fout.write(struct.pack("i" * n_dims, *shape[::-1]))
        fout.write(str)
        fout.seek((fout.tell() + 31) & -32)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: " + fname_out)
    print("")


if __name__ == '__main__':
    main()
