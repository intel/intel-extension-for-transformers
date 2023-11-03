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
import torch
import numpy as np
import re
import os
from pathlib import Path
import argparse
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Literal, Optional, Sequence, Tuple, TypeVar, Union)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

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
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"),
              ord("ÿ")+1))
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
    parser.add_argument("--outtype", choices=["f32", "f16"], default="fp32",
                        help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    dir_model = args.model.as_posix()
    fname_out = args.outfile.as_posix()

    # possible data types
    #   ftype == 0 -> float32, use_bf16 = False
    #   ftype == 1 -> float16, use_f16 = True
    use_f16 = False
    if args.outtype == "f16":
        use_f16 = True

    print("Loading model: ", dir_model)
    tokenizer = AutoTokenizer.from_pretrained(dir_model)
    config = AutoConfig.from_pretrained(dir_model, trust_remote_code=True)
    hparams = config.to_dict()
    model = AutoModelForCausalLM.from_pretrained(dir_model, config=config,
                                                 torch_dtype=torch.float16 \
                                                 if use_f16 else torch.float32,
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True)
    print("Model loaded: ", dir_model)

    list_vars = model.state_dict()

    encoder = tokenizer.vocab
    # Add added_tokens (special tokens) to the encoder
    encoder.update(tokenizer.get_added_vocab())
    print(hparams)

    print("Saving ne model to: ", fname_out)
    fout = open(fname_out, "wb")

    fout.write(struct.pack("i", 0x67676d6c)) # magic: ne in hex
    vocab_size = hparams["vocab_size"]
    fout.write(struct.pack("i", vocab_size))
    fout.write(struct.pack("i", hparams["n_embd"]))
    fout.write(struct.pack("i", hparams["n_positions"]))
    fout.write(struct.pack("i", hparams["n_head"]))
    fout.write(struct.pack("i", hparams.get("n_head_kv", 0)))  # multi-query attention
    fout.write(struct.pack("i", hparams["n_layer"]))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", use_f16))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("f", 0))
    fout.write(struct.pack("f", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    fout.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    
    fout.write(struct.pack("i", tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1))
    fout.write(struct.pack("i", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2))
    fout.write(struct.pack("i", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1))
    fout.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v:k for k, v in byte_encoder.items()}

    counter = 0
    # sort by value
    for key in sorted(encoder, key=encoder.get):
        text = bytearray([byte_decoder[c] for c in key])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        counter += 1

    # TODO: Repeat last token until vocab_size
    while counter < vocab_size:
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        counter += 1

    for name in list_vars.keys():
        data = list_vars[name].squeeze().numpy()
        print("Processing variable: " + name + " with shape: ", data.shape)

        # rename headers to keep compatibility
        if name == "transformer.ln_f.weight":
            name = "model/ln_f/g"
        elif name == "transformer.ln_f.bias":
            name = "model/ln_f/b"
        elif name == "transformer.wte.weight":
            name = "model/wte"
        elif name == "transformer.wpe.weight":
            name = "model/wpe"
        elif name == "lm_head.weight":
            name = "model/lm_head"
        elif re.match(r"transformer.h\.\d+\.ln_1\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/g"
        elif re.match(r"transformer.h\.\d+\.ln_1\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/b"
        elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/w"
        elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/b"
        elif re.match(r"transformer.h\.\d+\.attn\.c_proj\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/w"
        elif re.match(r"transformer.h.\d+.attn.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/b"
        elif re.match(r"transformer.h.\d+.ln_2.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/g"
        elif re.match(r"transformer.h.\d+.ln_2.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/b"
        elif re.match(r"transformer.h.\d+.mlp.c_fc.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/w"
        elif re.match(r"transformer.h.\d+.mlp.c_fc.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/b"
        elif re.match(r"transformer.h.\d+.mlp.c_proj.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/w"
        elif re.match(r"transformer.h.\d+.mlp.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/b"
        else:
            print("Unrecognized variable name. %s", name)

        n_dims = len(data.shape)

        ftype = 0
        if use_f16:
            if (name == "model/wte" or name == "model/lm_head" or name[-2:] == "/g"
                or name[-2:] == "/w") and n_dims == 2:
                print("  Converting to float16...")
                data = data.astype(np.float16)
                ftype = 1
            else:
                print("  Converting to float32...")
                data = data.astype(np.float32)
                ftype = 0

        # TODO NE MHA kernel without kv broadcast or repeat (when multi_query == True)
        if name[-14:] == "/attn/c_attn/w" or name[-14:] == "/attn/c_attn/b":
            print("  Duplicate K,V heads to use MHA instead of MQA")

            embed_dim = hparams["n_embd"]
            head_dim = embed_dim // hparams["n_head"]

            # ((n_heads + 2) * head_dim, hidden_dim) -> (3 * n_heads * head_dim, hidden_dim)
            q, k ,v = np.split(data, (hparams["n_head"] * head_dim,
                                    (hparams["n_head"] + 1) * head_dim), axis=0)
            # duplicate k, v along the first axis (head_dim, hidden_dim) ->
            #                                     (n_heads * head_dim, hidden_dim)
            if len(k.shape) == 2:
                k = np.tile(k, (hparams["n_head"], 1))
                v = np.tile(v, (hparams["n_head"], 1))
            elif len(k.shape) == 1:
                k = np.tile(k, (hparams["n_head"]))
                v = np.tile(v, (hparams["n_head"]))
            # concat q, k, v along the first axis (n_heads * head_dim, hidden_dim) ->
            #                                     (3 * n_heads * head_dim, hidden_dim)
            data = np.concatenate((q, k, v), axis=0)

        # header
        str = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: " + fname_out)
    print("")

if __name__ == "__main__":
    main()
