import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def transfer_8bit_to_4bit(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    assert input.shape[-1] % 2 == 0
    size = input.shape
    input = input.to(torch.int32)
    size = size[-1]
    size = int(size / 2)
    input[..., 0:size] = input[..., 0:size] + input[..., size:] * pow(2, 4)
    cache = input[..., 0:size].clone().to(torch.uint8)
    del input
    return cache


def transfer_4bit_to_8bit(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    size = input.shape
    low_end = input & 15
    #low_end = input % pow(2, 4)
    high_end = (input.to(torch.int32) - low_end.to(torch.int32)) / pow(2, 4)
    output = torch.cat((low_end, high_end), dim=-1).to(torch.uint8)
    return output

class CompressedUnion:
    def __init__(self, quantize_bit, compress_mode, group_size):
        self.quantize_bit = quantize_bit
        self.compress_mode = compress_mode
        self.group_size = group_size
        self.min = None
        self.step = None
        self.dtype = None
        self.shape = None
        self.is_compressed = False
        self.cache = None
        self.values = None
        self.counter = 0
        self.scale = None
        self.mn = None
        # self.kvcache_shape = None

    def set_cache(self, input: torch.Tensor):
        self.counter += 1
        # has_inf = torch.isinf(input)
        # has_nan = torch.isnan(input)
        # print(self.counter,has_inf.any(),has_nan.any())
        self.cache = input
        self.shape = input.shape
        self.kvcache_shape = input.shape

    def get_cache(self):
        return self.cache

    def clean_cache(self):
        self.is_compressed = False
        self.cache = None
        self.values = None
        self.min = None
        self.step = None
        #self.shape = None
        self.scale = None

    def compress(self):
        input = self.cache
        self.dtype = input.dtype
        self.is_compressed = True

        if self.compress_mode == "channel_asymmetric_quantization":
            self.cache, self.scale, self.mn = channel_asymmetric_quantization(input, self.quantize_bit, self.shape, self.group_size)
        elif self.compress_mode == "token_asymmetric_quantization":
            self.cache, self.scale, self.mn = token_asymmetric_quantization(input, self.quantize_bit, self.shape, self.group_size)

    def decompress(self):
        self.is_compressed = False
        if self.compress_mode == "channel_asymmetric_quantization":
            output = channel_asymmetric_dequantization(
                self.cache, self.scale, self.mn, self.quantize_bit, self.shape, self.dtype, self.group_size)
        elif self.compress_mode == "token_asymmetric_quantization":
            output = token_asymmetric_dequantization(
                self.cache, self.scale, self.mn, self.quantize_bit, self.shape, self.dtype, self.group_size)
        #self.clean_cache()
        return output

def channel_asymmetric_quantization(
    input: torch.Tensor, quantize_bit, shape, group_size=128
):
    batch, num_head, seq_len, sep_dim = shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    mx, mn = input.max(dim=-2)[0], input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)

    scale = (mx - mn) / (2**quantize_bit - 1)
    quantized_input = (input - mn) / scale
    quantized_input = F.relu(quantized_input)

    quantized_input = quantized_input.to(torch.uint8)
    quantized_input = transfer_8bit_to_4bit(quantized_input)
    return quantized_input, scale, mn

def token_asymmetric_quantization(
    input: torch.Tensor, quantize_bit, shape, group_size=128
):
    batch, num_head, seq_len, sep_dim = shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    num_groups = (sep_dim * num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / (2**quantize_bit - 1)
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)

    input_in_groups = input_in_groups.to(torch.uint8)
    input_in_groups = transfer_8bit_to_4bit(input_in_groups)
 
    return input_in_groups, scale, mn

def channel_asymmetric_dequantization(
    input: torch.Tensor, scale, mn, quantize_bit, shape, dtype, group_size=128
):
    input = transfer_4bit_to_8bit(input)

    input = input.type(dtype)
    input = input * scale + mn
    output = input.reshape(shape)
    return output

def token_asymmetric_dequantization(
    input: torch.Tensor, scale, mn, quantize_bit, shape, dtype, group_size=128
):
    input = transfer_4bit_to_8bit(input)

    input = input.type(dtype)
    input = input * scale + mn
    output = input.reshape(shape)
    return output

class Buffer:
    def __init__(self, is_k_states=True):
        self.cache = None
        self.inp_seq_len = -1
        self.compress_cache = False
        self.is_k_states = is_k_states

    def compress(self, input):
        self.cache.set_cache(input)
        self.cache.compress()
        return self.cache

    def get_cache(self):
        return self.cache.decompress()

    def build_compress_env(self):
        config_path = "./quantization_config/kv_cache.json"
        # config_path = None
        if config_path is not None:
            import json
            self.compress_cache = True
            with open(config_path) as config_json:
                compress_kwargs = json.load(config_json)
            if self.is_k_states:
                self.quantize_bit = compress_kwargs["k_quantize_bit"]
                self.compress_mode = compress_kwargs["k_compress_mode"]
                self.group_size = compress_kwargs["k_group_size"]
            else:
                self.quantize_bit = compress_kwargs["v_quantize_bit"]
                self.compress_mode = compress_kwargs["v_compress_mode"]
                self.group_size = compress_kwargs["v_group_size"]
            self.cache = CompressedUnion(self.quantize_bit, self.compress_mode, self.group_size)

    def set_cache(self, prev_shape, cur, dim, idx, inp_seq_len):
        #prev = prev.decompress()
        prev = self.cache.decompress()
        if prev.shape[2] < prev_shape[2]:
            pad_amount = prev_shape[2] - prev.shape[2]
            prev = torch.nn.functional.pad(prev, (0, 0, 0, pad_amount), value=0)

        if prev.shape == cur.shape:
            prev.copy_(cur)
        elif cur.shape[2] > 1 and cur.shape[2] <= prev.shape[2]:
            # Initialize
            prev[:, :, :inp_seq_len, :].copy_(cur)
        elif idx is not None:
            assert cur.shape[2] == 1, f"Cannot update kv-cache. Unsupported shapes. prev:{prev.shape} cur:{cur.shape}"
            prev.index_copy_(dim, idx - 1, cur)
        else:
            prev = torch.cat((prev, cur), dim=dim)
        self.cache.set_cache(prev)
        self.cache.compress()
        return self.cache

