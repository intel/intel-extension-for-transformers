import os
import json
import sys
import re
import argparse
from common import *

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
        out1 = permute_func(weight.view(-1,weight.shape[-1]).t(), 32, 32)
    else:
        out1 = weight.view(-1,weight.shape[-1]).t()
    tensor = out1.reshape(-1, 32) #+ 8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)

    if permute:
        out2 = permute_func(scales.view(-1,scales.shape[-1]).t(), 32, 32)
    else:
        out2 = scales.view(-1,scales.shape[-1]).t()
    gptq_scale = out2.reshape(-1,1)
    gptq_scale = torch.cat([gptq_scale,gptq_scale,gptq_scale,gptq_scale], dim=1).view(-1,1)
    pack_tensor = torch.cat((gptq_scale.half().view(torch.int8), tensor), dim=-1)

    weight = (scales * (weight - zeros))
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    return weight.t(), pack_tensor

def write_header(fout, shape, dst_name, ftype_cur):
    sname = dst_name.encode('utf-8')
    fout.write(struct.pack("iii", len(shape), len(sname), ftype_cur))
    fout.write(struct.pack("i" * len(shape), *shape[::-1]))
    fout.write(sname)
    fout.seek((fout.tell() + 31) & -32)

def convert_fp32_tensor(src_name, dst_name, model, fout):
    v = model[src_name]
    shape = v.shape
    # print("Processing non-Q4 variable: " + src_name +
    #       " with shape: ", shape, " and type: ", v.dtype)
    v = v.to(torch.float32)

    ftype_cur = {torch.float16: 1, torch.float32: 0}[v.dtype]

    # header
    write_header(fout, shape, dst_name, ftype_cur)

    # data
    v.numpy().tofile(fout)
    print(f"converting {dst_name} float tensor")

def convert_q4_tensor(src_name, dst_name, model, fout, n_head, n_head2=0, permute=False):
    qzeros = model[f"{src_name}.qzeros"]
    zeros = qzeros_to_zeros(qzeros)
    scales = model[f"{src_name}.scales"]
    g_idx = model[f"{src_name}.g_idx"]
    qweight = model[f"{src_name}.qweight"]

    weight, pack = recover_weight(qweight, scales, qzeros, permute)
    
    shape = weight.shape
    # weight = weight.to(torch.float32)
    write_header(fout, shape, dst_name, 2)
    pack.numpy().tofile(fout)
    print(f"converting {dst_name} qauntized tensor to ggml q4 block")

def find_quantized_model_file(model_path):
    model_path = Path(model_path)
    for ext in ['.safetensors', '.pt']:
        found = list(model_path.glob(f"*{ext}"))
        if len(found) > 0:
            if len(found) != 1:
                warnings.warn(f'Detected {len(found)} {ext} model, use the first one {found[0]}.')
            print(f"Detected model file {found[0]}")
            return str(found[0])

def load_gptq_model(model_path):
    input_path = find_quantized_model_file(model_path)
    model = None
    if input_path.endswith('pt'):
        model = torch.load(input_path, map_location="cpu")
    elif input_path.endswith('safetensors'):
        from safetensors.torch import load_file
        model = load_file(input_path)
    else:
        print("unknown input model path, only support .safetensors or .pt file.")
    return model

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype", choices=["f32", "f16"], help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)

    out_path = args.outfile.as_posix()
    model_path = args.model.as_posix()

    model = load_gptq_model(model_path)
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
    convert_fp32_tensor("model.embed_tokens.weight", "tok_embeddings.weight", list_vars, f)
    convert_fp32_tensor("model.norm.weight", "norm.weight", list_vars, f)
    convert_fp32_tensor("lm_head.weight", "output.weight", list_vars, f)

    for i in range(n_layer):
        convert_q4_tensor(f"model.layers.{i}.self_attn.q_proj",
                    f"layers.{i}.attention.wq.weight", list_vars, f, n_head, n_head, permute=True)
        convert_q4_tensor(f"model.layers.{i}.self_attn.k_proj",
                    f"layers.{i}.attention.wk.weight", list_vars, f, n_head, n_head_kv, permute=True)
        convert_q4_tensor(f"model.layers.{i}.self_attn.v_proj",
                    f"layers.{i}.attention.wv.weight", list_vars, f, n_head)
        convert_q4_tensor(f"model.layers.{i}.self_attn.o_proj",
                    f"layers.{i}.attention.wo.weight", list_vars, f, n_head)
        convert_q4_tensor(f"model.layers.{i}.mlp.gate_proj",
                    f"layers.{i}.feed_forward.w1.weight", list_vars, f, n_head)
        convert_q4_tensor(f"model.layers.{i}.mlp.down_proj",
                    f"layers.{i}.feed_forward.w2.weight", list_vars, f, n_head)
        convert_q4_tensor(f"model.layers.{i}.mlp.up_proj",
                    f"layers.{i}.feed_forward.w3.weight", list_vars, f, n_head)

        convert_fp32_tensor(f"model.layers.{i}.input_layernorm.weight",
                        f"layers.{i}.attention_norm.weight", list_vars, f)
        convert_fp32_tensor(f"model.layers.{i}.post_attention_layernorm.weight",
                        f"layers.{i}.ffn_norm.weight", list_vars, f)


    f.close()
    print(f"Success! saved as {out_path}")

if __name__ == '__main__':
    main()