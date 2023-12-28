
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
import sys
import struct
import json
import code
import torch
import numpy as np
from pathlib import Path
import argparse
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypeVar,
                    Union)
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
#model = AutoModelForCausalLM.from_pretrained("./", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
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

    # possible data types
    #   ftype == 0 -> float32
    #   ftype == 1 -> float16
    ftype = 0
    if args.outtype == "f16":
        ftype = 1

    tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
 #   import pdb;pdb.set_trace()
    print("Loading model: ", dir_model)
    model = AutoModelForCausalLM.from_pretrained("./", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained(dir_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hparams = model.config.to_dict()
    print("Model loaded: ", dir_model)

    fout = open(fname_out, "wb")

    # 0x67676d6c is unversioned ne
    # 0x67676d66 is versioned ggmf (requires token scores)
    ne_file_magic = 0x67676d66
    #ne_file_version = 0x00000001 # v1

    fout.write(struct.pack("i", ne_file_magic))  # magic: ne in hex
    fout.write(struct.pack("i", 1))
    #import pdb;pdb.set_trace()
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["hidden_size"]))
    fout.write(struct.pack("i", hparams["intermediate_size"]))  # dummy data
    fout.write(struct.pack("i", hparams["num_attention_heads"]))
    fout.write(struct.pack("i", hparams["num_key_value_heads"]))  # multi-query attention
    fout.write(struct.pack("i", hparams["num_key_value_heads"]))
    fout.write(struct.pack("i", hparams["num_key_value_heads"]))
    fout.write(struct.pack("i", ftype))
    fout.write(struct.pack("i", hparams["max_sequence_length"]))
    fout.write(struct.pack("f", 0.0))
    fout.write(struct.pack("f", 0.0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
    fout.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("f", hparams.get("rms_norm_eps", 1e-6)))  # rms norm eps
    fout.write(struct.pack("f", 10000.0))  # freq_base
    fout.write(struct.pack("f", 1.0))  # rope_factor
    
    fout.write(struct.pack("i", int(hparams["rope_scaling"]["factor"])))
    # import pdb;pdb.set_trace()
    fout.write(struct.pack("i", hparams["rope_scaling"]["original_max_position_embeddings"]))
    fout.write(struct.pack("i", 1 if hparams["rope_scaling"]["type"]=="yarn" else 0))

    fout.write(struct.pack("i", hparams["bos_token_id"]))
    fout.write(struct.pack("i", hparams["eos_token_id"]))
    fout.write(struct.pack("i", hparams["pad_token_id"]))
    fout.write(struct.pack("i", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1))
    for i in range(hparams["vocab_size"]):
        if i < tokenizer.vocab_size:
            text = tokenizer.decode([i]).encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("f", 0.0 - i))
        else:
            text = tokenizer.decode([tokenizer.vocab_size - 1]).encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
            fout.write(struct.pack("f", -10000))

    list_vars = model.state_dict()

    print(hparams)

    for name in list_vars.keys():
        # No gradients for these
        list_vars[name].requires_grad = False
        src = name
        nn = name
        print(src, ' -> ', name)
        list_vars[src]=list_vars[src].float()
        data = list_vars[src].squeeze().numpy()
        data = data.astype(np.float32)

        n_dims = len(data.shape)
        print(name, n_dims, data.shape)

        # default type is fp32
        ftype_cur = 0
        if ftype == 1 and n_dims > 1:
            print("  Converting to float16", data.shape, data[:3, :3].tolist())
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32", data.shape, data[:3, :3].tolist() if n_dims > 1 else data[:3].tolist())
            data = data.astype(np.float32)

        # header
        str = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        print(str)
        fout.write(str)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: " + fname_out)
    print("")


if __name__ == '__main__':
    main()
