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
import os
import numpy as np
from pathlib import Path
import argparse
from typing import List, Optional
import subprocess
import struct
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_maps = {"gpt_neox": "gptneox", "RefinedWebModel": "falcon"}

class ConvertModel:
    def __init__(self, out_file, dir_model, with_vocab=True):
        self.out_file = out_file
        self.with_vocab = with_vocab
        self.dir_model = dir_model
        self.ftype = 0
        self.fout = None
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.dir_model, low_cpu_mem_usage=True, trust_remote_code=True)
        self.hparams = self.model.config.to_dict()

    def write_head(self):
        self.fout.write(struct.pack("i", 0x67676D6C))

    def write_params(self):
        pass

    def write_weight(self):
        list_vars = self.model.state_dict()
        for name in list_vars.keys():
            print(name, list_vars[name].shape, list_vars[name].dtype)
        for name in list_vars.keys():
            data = list_vars[name].squeeze().numpy()
            print("Processing variable: " + name + " with shape: ", data.shape)

            n_dims = len(data.shape)

            # ftype == 0 -> float32, ftype == 1 -> float16
            ftype_cur = 0
            if self.ftype != 0:
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
            self.fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
            for i in range(n_dims):
                self.fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
            self.fout.write(str)

            # data
            data.tofile(self.fout)
    def write_vocab(self):
        pass

    def convert(self):
        self.fout = open(self.out_file, "wb")
        self.load_model()
        self.write_head()
        self.write_params()
        if self.with_vocab:
            self.write_vocab()
        self.write_weight()

        self.fout.close()

class ConvertMPT(ConvertModel):
    def __init__(self, out_file, dir_model, with_vocab=True):
        super().__init__(out_file, dir_model, with_vocab)

    def write_params(self):
        self.fout.write(struct.pack("i", self.hparams["vocab_size"]))
        self.fout.write(struct.pack("i", self.hparams["d_model"]))
        self.fout.write(struct.pack("i", self.hparams["d_model"]))
        self.fout.write(struct.pack("i", self.hparams["n_heads"]))
        self.fout.write(struct.pack("i", self.hparams.get("n_head_kv", 0)))  # multi-query attention
        self.fout.write(struct.pack("i", self.hparams["n_layers"]))
        self.fout.write(struct.pack("i", self.hparams["n_layers"]))
        self.fout.write(struct.pack("i", self.ftype))
        self.fout.write(struct.pack("i", self.hparams["max_seq_len"]))
        self.fout.write(struct.pack("f", self.hparams["attn_config"]["alibi_bias_max"]))
        self.fout.write(struct.pack("f", self.hparams["attn_config"]["clip_qkv"] or 0.0))
        self.fout.write(struct.pack("i", 0))
        self.fout.write(struct.pack("i", 0))  # word_embed_proj_dim (for opt)
        self.fout.write(struct.pack("i", 0))  # do_layer_norm_before (for opt)

        self.fout.write(struct.pack("i", 0))
        self.fout.write(struct.pack("i", 0))
        self.fout.write(struct.pack("i", 0))

        self.fout.write(struct.pack("i", int(-1 if (self.hparams.get("bos_token_id", -1)) is None else (self.hparams.get("bos_token_id", -1)))))
        self.fout.write(struct.pack("i", int(-1 if (self.hparams.get("eos_token_id", -1)) is None else (self.hparams.get("eos_token_id", -1)))))
        self.fout.write(struct.pack("i", 0))
        self.fout.write(struct.pack("i", 0))

    def write_vocab(self):
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
        vocab_size = self.hparams["vocab_size"]

        encoder = self.tokenizer.vocab
        # Add added_tokens (special tokens) to the encoder
        encoder.update(self.tokenizer.get_added_vocab())

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
            self.fout.write(struct.pack("i", len(text)))
            self.fout.write(text)
            counter += 1

        # Repeat last token until vocab_size
        while counter < vocab_size:
            self.fout.write(struct.pack("i", len(text)))
            self.fout.write(text)
            counter += 1

def convert_model(model, outfile, outtype):
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_type = model_maps.get(config.model_type, config.model_type)

    path = Path(Path(__file__).parent.absolute(), "convert_{}.py".format(model_type))
    cmd = []
    cmd.extend(["python", path])
    cmd.extend(["--outfile", outfile])
    cmd.extend(["--outtype", outtype])
    cmd.extend([model])

    print("cmd:", cmd)
    subprocess.run(cmd)

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch model to a NE compatible file"
    )
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16"],
        help="output format, default: f32",
        default="f32",
    )
    parser.add_argument("--outfile", type=Path, required=True, help="path to write to")
    parser.add_argument(
        "model", type=Path, help="directory containing model file or model id"
    )
    args = parser.parse_args(args_in)

    if args.model.exists():
        dir_model = args.model.as_posix()
    else:
        dir_model = args.model

    model = ConvertMPT(args.outfile, dir_model)
    model.convert()


if __name__ == "__main__":
    main()
