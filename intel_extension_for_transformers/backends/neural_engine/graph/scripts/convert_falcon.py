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

import io
import os
import sys
import struct
import json
import code
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

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

if len(sys.argv) < 3:
    print("Usage: python convert_falcon.py model_name dir-output [use-f32]")
    print("  model_name: name of the model to convert. Example: 'falcon-7b'")
    print("  dir-output: directory where the output file will be written")
    print("  use-f32:    if present, use float32 instead of float16")
    sys.exit(1)

model_name = sys.argv[1]
dir_out = sys.argv[2]

# make sure the output directory exists
os.makedirs(dir_out, exist_ok=True)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]
ftype = 1
if len(sys.argv) > 3:
    ftype = 0

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
hparams = config.to_dict()
print("Loading model: ", model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16
                if ftype == 1 else torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
print("Model loaded: ", model_name)


fname_out = os.path.join(dir_out, f"ne-{ftype_str[ftype]}.bin")
fout = open(fname_out, "wb")
fout.write(struct.pack("i", 0x67676d6c)) # magic: falcon in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", ftype))

reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}
byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

for i in range(hparams["vocab_size"]):
    text = bytearray([byte_decoder[c] for c in reverse_vocab[i]])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

list_vars = model.state_dict()
for name in list_vars.keys():
    src = name
    data = list_vars[src].squeeze().numpy()
    data = data.astype(np.float32)

    n_dims = len(data.shape)
    print(name, n_dims, data.shape)

    # default type is fp32
    ftype_cur = 0
    if ftype == 1 and n_dims > 1:
        print("  Converting to float16")
        data = data.astype(np.float16)
        ftype_cur = 1

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
