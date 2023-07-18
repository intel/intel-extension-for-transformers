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
import sys
import struct
import json
import numpy as np
from pathlib import Path
import argparse
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Literal, Optional, Sequence, Tuple, TypeVar, Union)
from transformers import AutoModelForCausalLM, AutoTokenizer
import sentencepiece.sentencepiece_model_pb2 as model

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


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    dir_model = args.model.as_posix()
    fname_out = args.outfile.as_posix()

    with open(dir_model + '/config.json', "r", encoding="utf-8") as f:
        hparams = json.load(f)

    # possible data types
    #   ftype == 0 -> float32
    #   ftype == 1 -> float16
    ftype = 0
    if args.outtype== "f16":
        ftype = 1


    tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        dir_model, low_cpu_mem_usage=True, trust_remote_code=True
    )


    list_vars = model.state_dict()
    for name in list_vars.keys():
        print(name, list_vars[name].shape, list_vars[name].dtype)

    fout = open(fname_out, "wb")

    print(hparams)

    fout.write(struct.pack("i", 0x67676D6C))
    fout.write(struct.pack("i", hparams["d_model"]))
    fout.write(struct.pack("i", hparams["max_seq_len"]))
    fout.write(struct.pack("i", hparams["n_heads"]))
    fout.write(struct.pack("i", hparams["n_layers"]))
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("f", hparams["attn_config"]["alibi_bias_max"]))
    fout.write(struct.pack("f", hparams["attn_config"]["clip_qkv"] or 0.0))
    fout.write(struct.pack("i", ftype))

    vocab_size = hparams["vocab_size"]

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
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        counter += 1

    # Repeat last token until vocab_size
    while counter < vocab_size:
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        counter += 1

    # assert counter == config.vocab_size

    for name in list_vars.keys():
        data = list_vars[name].squeeze().numpy()
        print("Processing variable: " + name + " with shape: ", data.shape)

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

        # header
        str = name.encode("utf-8")
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: " + fname_out)
    print("")


if __name__ == '__main__':
    main()
