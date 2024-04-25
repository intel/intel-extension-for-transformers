import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def fake_groupwise_channel_asymmetric_quantization(
    input: torch.Tensor, quantize_bit, group_size=128
):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    group_size = 128
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    mx, mn = input.max(dim=-2)[0], input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)

    scale = (mx - mn) / (2**quantize_bit - 1)
    quantized_input = (input - mn) / scale
    quantized_input = F.relu(quantized_input)
    rounded_input = quantized_input.round_()
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch, seq_len, num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_groupwise_token_asymmetric_quantization(
    input: torch.Tensor, quantize_bit, group_size=128
):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    group_size = 128
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
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input


def compress_insert_function(
    previous_key,
    previous_value,
    compress_config,
    pbase1=None,
    qbase1=None,
    pbase2=None,
    qbase2=None,
    attn_weights=None,
):
    batch, num_head, seq_len, sep_dim = previous_key.shape
    starting_idx = int(0)
    locality_idx = -seq_len
    # print("starting_idx:", starting_idx, "locality_idx:", locality_idx,compress_config.token_preserving[layer_idx],batch, num_head, seq_len, sep_dim)

    # SK: TODO take a look at this: idea merged from KIVI
    if compress_config["compress_method"] == "groupquantization_kc_vt":
        previous_key[:, :, starting_idx:-locality_idx, :] = (
            fake_groupwise_channel_asymmetric_quantization(
                previous_key[:, :, starting_idx:-locality_idx, :],
                compress_config["quantize_bit"],
                128,
            )
        )
        if previous_value is not None:
            previous_value[:, :, starting_idx:-locality_idx, :] = (
                fake_groupwise_token_asymmetric_quantization(
                    previous_value[:, :, starting_idx:-locality_idx, :],
                    compress_config["quantize_bit"],
                    128,
                )
            )
    return previous_key, previous_value


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
        self.shape = None
        self.scale = None

    def compress(self):
        # print("compress")
        input = self.cache
        self.dtype = input.dtype
        self.is_compressed = True
        if self.compress_mode == "channel_asymmetric_quantization":
            self.cache, self.scale, self.mn = channel_asymmetric_quantization(input, self.quantize_bit, self.shape, self.group_size)
        elif self.compress_mode == "token_asymmetric_quantization":
            self.cache, self.scale, self.mn = token_asymmetric_quantization(input, self.quantize_bit, self.shape, self.group_size)

    def decompress(self):
        # print("decompress")
        self.is_compressed = False
        if self.compress_mode == "channel_asymmetric_quantization":
            output = channel_asymmetric_dequantization(
                self.cache, self.scale, self.mn, self.quantize_bit, self.shape, self.group_size).to(self.dtype)
        elif self.compress_mode == "token_asymmetric_quantization":
            output = token_asymmetric_dequantization(
                self.cache, self.scale, self.mn, self.quantize_bit, self.shape, self.group_size).to(self.dtype)
        # self.clean_cache()
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
    rounded_input = quantized_input.round_().to(torch.int32)

    blob_size = (rounded_input.shape[2] - 1) // (32 // quantize_bit) + 1
    q_input = torch.zeros((rounded_input.shape[0], rounded_input.shape[1], blob_size), dtype=torch.int32, device=rounded_input.device)
    # for i in range(blob_size):
    #     for j in range(rounded_input.shape[2]):
    #         q_input[:, :, i].__ior__(rounded_input[:, :, j].__ilshift__(quantize_bit * j))
    idx =  torch.tensor(list(range(0, 32, quantize_bit)), dtype=torch.int32).unsqueeze(0)
    step = 32 // quantize_bit
    shifted = torch.bitwise_left_shift(rounded_input, idx.repeat(1, rounded_input.shape[2] // step))
    for i in range(blob_size):
        q_input[:, :, i] = torch.sum(shifted[:, :, i * step:(i+1) * step], dim=-1)
    # print("compress done")
    return q_input, scale, mn

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
    rounded_input = input_in_groups.round_().to(torch.int32).view(batch, seq_len, sep_dim * num_head)

    blob_size = (rounded_input.shape[2] - 1) // (32 // quantize_bit) + 1
    q_input = torch.zeros((rounded_input.shape[0], rounded_input.shape[1], blob_size), dtype=torch.int32, device=rounded_input.device)
    # for i in range(blob_size):
    #     for j in range(rounded_input.shape[2]):
    #         q_input[:, :, i].__ior__(rounded_input[:, :, j].__ilshift__(quantize_bit * j))
    idx =  torch.tensor(list(range(0, 32, quantize_bit)), dtype=torch.int32).unsqueeze(0)
    step = 32 // quantize_bit
    shifted = torch.bitwise_left_shift(rounded_input, idx.repeat(1, rounded_input.shape[2] // step))
    for i in range(blob_size):
        q_input[:, :, i] = torch.sum(shifted[:, :, i * step:(i+1) * step], dim=-1)
    # print("compress done")
    return q_input, scale, mn

def channel_asymmetric_dequantization(
    input: torch.Tensor, scale, mn, quantize_bit, shape, group_size=128
):
    batch, num_head, seq_len, sep_dim = shape
    # rounded_input = torch.zeros((batch, seq_len, sep_dim * num_head), dtype=torch.int32, device=input.device)
    # for i in range(input.shape[-1]):
    #     for j in range(rounded_input.shape[2]):
    #         rounded_input[:, :, j].__ior__(rounded_input[:, :, i].__irshift__(quantize_bit * (rounded_input.shape[2] - j)))
    # import pdb;pdb.set_trace()
    rounded_input = input.repeat(1,1,sep_dim * num_head // input.shape[2])
    idx =  torch.tensor(list(range(0, 32, quantize_bit)), dtype=torch.int32).unsqueeze(0)
    rounded_input = torch.bitwise_right_shift(rounded_input, idx.repeat(1, rounded_input.shape[2] // (32 // quantize_bit)))

    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch, seq_len, num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    # print("decompress done")
    return dequantized_input



def token_asymmetric_dequantization(
    input: torch.Tensor, scale, mn, quantize_bit, shape, group_size=128
):
    batch, num_head, seq_len, sep_dim = shape
    # rounded_input = torch.zeros((batch, seq_len, sep_dim * num_head), dtype=torch.int32, device=input.device)
    # for i in range(input.shape[-1]):
    #     for j in range(rounded_input.shape[2]):
    #         rounded_input[:, :, j].__ior__(rounded_input[:, :, i].__irshift__(quantize_bit * (rounded_input.shape[2] - j)))
    rounded_input = input.repeat(1,1,sep_dim * num_head // input.shape[2])
    idx =  torch.tensor(list(range(0, 32, quantize_bit)), dtype=torch.int32).unsqueeze(0)
    rounded_input = torch.bitwise_right_shift(rounded_input, idx.repeat(1, rounded_input.shape[2] // (32 // quantize_bit)))

    rounded_input = rounded_input.view(batch, seq_len, num_head, sep_dim)
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    # print("decompress done")
    return dequantized_input