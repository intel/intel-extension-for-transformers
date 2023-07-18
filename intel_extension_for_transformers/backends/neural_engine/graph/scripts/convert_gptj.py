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
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Literal, Optional, Sequence, Tuple, TypeVar, Union)
from transformers import GPTJForCausalLM

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

    ftype = 0
    if args.outtype== "f16":
        ftype = 1

    # output in the same directory as the model
    with open(dir_model + "/vocab.json", "r", encoding="utf-8") as f:
        encoder = json.load(f)
    
    with open(dir_model + "/added_tokens.json", "r", encoding="utf-8") as f:
        encoder_added = json.load(f)
    
    with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
        hparams = json.load(f)

    print("Loading model: ", dir_model)
    model = GPTJForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True)
    list_vars = model.state_dict()
    fout = open(fname_out, "wb")
    
    fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["n_positions"]))
    fout.write(struct.pack("i", hparams["n_embd"]))
    fout.write(struct.pack("i", hparams["n_head"]))
    fout.write(struct.pack("i", hparams["n_layer"]))
    fout.write(struct.pack("i", hparams["rotary_dim"]))
    fout.write(struct.pack("i", ftype))
    
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v:k for k, v in byte_encoder.items()}
    
    fout.write(struct.pack("i", len(encoder) + len(encoder_added)))
    
    for key in encoder:
        text = bytearray([byte_decoder[c] for c in key])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
    
    for key in encoder_added:
        text = bytearray([byte_decoder[c] for c in key])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
    
    for name in list_vars.keys():
        data = list_vars[name].squeeze().numpy()
        print("Processing variable: " + name + " with shape: ", data.shape)
    
        # we don't need these
        if name.endswith("attn.masked_bias") or name.endswith(".attn.bias"):
            print("  Skipping variable: " + name)
            continue
    
        n_dims = len(data.shape);
    
        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype_cur = 0;
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
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str);
    
        # data
        data.tofile(fout)
    
    fout.close()

    print("Done. Output file: " + fname_out)
    print("")

if __name__ == '__main__':
    main()
