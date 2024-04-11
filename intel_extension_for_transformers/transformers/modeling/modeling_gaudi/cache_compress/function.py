import torch
import time


def transfer_8bit_to_4bit(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    assert input.shape[-1] % 2 == 0
    size = input.shape
    size = size[-1]
    size = int(size / 2)
    input[0:size] = input[0:size] + input[size:] * pow(2, 4)
    cache = input[0:size].clone()
    del input
    return cache


def transfer_4bit_to_8bit(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    size = input.shape
    low_end = input % pow(2, 4)
    high_end = (input - low_end) / pow(2, 4)
    output = torch.cat((low_end, high_end), dim=0)
    return output


def transfer_8bit_to_4bit_batchwise(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    assert input.shape[-1] % 2 == 0
    size = input.shape
    size = size[-1]
    size = int(size / 2)
    input[..., 0:size] = input[..., 0:size] + input[..., size:] * pow(2, 4)
    cache = input[..., 0:size].clone()
    # del input
    return cache


def transfer_4bit_to_8bit_batchwise(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    size = input.shape
    low_end = input % pow(2, 4)
    high_end = (input - low_end) / pow(2, 4)
    output = torch.cat((low_end, high_end), dim=-1)
    return output


def true_uniform_quantization_compress(input: torch.Tensor, quantize_bit):
    if quantize_bit != 8 and quantize_bit != 4:
        raise ValueError("quantize_bit should be 8 or 4")
    shape = input.shape
    bsz = shape[0]
    input = input.reshape(-1)
    if quantize_bit == 8:
        input = input.float()  # convert to 32bits to avoid max - min = inf
    min, max = input.min(), input.max()
    # step = (max - min) / (pow(2, quantize_bit) - 1)
    scale = (max - min) / (2**quantize_bit - 1)
    # print("before min max:",min,max,step)
    quantized_input = (input - min) / scale
    # print("after min max:",quantized_input.min(),quantized_input.max())
    # print("quantized isnan:",torch.any(torch.isnan(quantized_input)))
    quantized_input = quantized_input.round_()
    quantized_input = quantized_input.to(torch.uint8)
    if quantize_bit == 4:
        quantized_input = transfer_8bit_to_4bit(quantized_input)
    # print("isnan:",torch.any(torch.isnan(returning_input)))
    # while(True):
    #     pass
    return quantized_input, shape, min, scale


def true_uniform_quantization_decompress(
    input: torch.Tensor, quantize_bit, shape, min, step, dtype
):
    if quantize_bit != 8 and quantize_bit != 4:
        raise ValueError("quantize_bit should be 8 or 4")
    input = input.reshape(-1)
    if quantize_bit == 8:
        input = input.float()
        input = input * step + min
        output = input.reshape(shape).type(dtype)
    elif quantize_bit == 4:
        input = transfer_4bit_to_8bit(input)

        input = input.type(dtype)
        input = input * step + min
        output = input.reshape(shape)
    return output


def true_outlier_quantization_compress(input: torch.Tensor, quantize_bit, left):
    shape = input.shape
    input = input.reshape(-1)
    left_num = int(len(input) * left / 2)
    value1, indices1 = torch.topk(input, left_num, largest=False)
    value2, indices2 = torch.topk(input, left_num, largest=True)
    values = torch.cat((value1, value2), dim=0)
    indices = torch.cat((indices1, indices2), dim=0)

    input = input.index_fill_(0, indices, 0)
    output, _, min, step = true_uniform_quantization_compress(input, quantize_bit)

    return output, shape, min, step, values, indices


def true_outlier_quantization_decompress(
    input: torch.Tensor, quantize_bit, shape, min, step, dtype, values, indices
):
    input = true_uniform_quantization_decompress(
        input, quantize_bit, shape, min, step, dtype
    )
    input = input.reshape(-1)
    input[indices] = values
    input = input.reshape(shape)
    return input


def fake_quant_error_simulation(input: torch.Tensor, quantize_bit):
    input = input.reshape(-1)

    min, max = input.min(), input.max()
    step = (max - min) / (pow(2, quantize_bit) - 1)
    # print("before min max:",min,max,step)
    error = input - torch.round((input - min) / step)
    return error, min, step


def true_poweriteration(input: torch.Tensor, loop, rank, p_base=None, q_base=None):
    # input size [batch,num_head,seq_len,model_dim/num_head]
    # -> [batch,seq_len,model_dim] -> [batch * seq_len,model_dim]
    # p_base = torch.rand(input.shape[3] * input.shape[1], rank).to(device)
    # q_base = torch.rand(input.shape[0] * input.shape[2], rank).to(device)
    batch, num_head, seq_len, sep_dim = input.shape
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )  # convert to 32bits for qr decomposition
    input = input.view(batch, seq_len, sep_dim * num_head)
    input = input.float()
    if q_base is not None and p_base is not None:
        p_base[0] = p_base[0].float()
        q_base[0] = q_base[0].float()
    else:
        p_base = [torch.rand(batch, sep_dim * num_head, rank).to(input.device).float()]
        q_base = [torch.rand(batch, seq_len, rank).to(input.device).float()]
    # 3 calculation = loop * (matmul) + 2 * qrO(n^2)
    for i in range(loop):
        if i == loop - 1:
            p_base[0] = torch.linalg.qr(p_base[0].float()).Q
        q_base[0] = input @ p_base[0]
        if i == loop - 1:
            q_base[0] = torch.linalg.qr(q_base[0].float()).Q
        p_base[0] = torch.transpose(input, 1, 2) @ q_base[0]
    # input = q_base[0] @ torch.transpose(p_base[0], 0, 1)
    # input = input.view(batch, seq_len, num_head, sep_dim)
    # input = input.permute(0, 2, 1, 3)
    # input = input.type(torch.bfloat16)
    p_base[0] = p_base[0].half()
    q_base[0] = q_base[0].half()
    return p_base, q_base


def true_gear_compress(input: torch.Tensor, quantize_bit, left, rank, loop):
    shape = input.shape
    input = input.reshape(-1)
    left_num = int(len(input) * left / 2)
    value1, indices1 = torch.topk(input, left_num, largest=False)
    value2, indices2 = torch.topk(input, left_num, largest=True)
    values = torch.cat((value1, value2), dim=0)
    indices = torch.cat((indices1, indices2), dim=0)
    input = input.index_fill_(0, indices, 0)
    error, min, step = fake_quant_error_simulation(input, quantize_bit)
    error = error.index_fill_(0, indices, 0)
    error = error.reshape(shape)
    p_base, q_base = true_poweriteration(error, loop, rank)
    # has_inf = torch.isinf(p_base[0])
    # has_nan = torch.isnan(p_base[0])
    # if has_inf.any() or has_nan.any():
    #     print("pbase",has_inf.any(),has_nan.any())
    # has_inf = torch.isinf(q_base[0])
    # has_nan = torch.isnan(q_base[0])
    # if has_inf.any() or has_nan.any():
    #     print("qbase",has_inf.any(),has_nan.any())
    output, _, min, step = true_uniform_quantization_compress(input, quantize_bit)
    return output, shape, min, step, values, indices, p_base, q_base


def true_gear_decompress(
    input: torch.Tensor,
    quantize_bit,
    shape,
    min,
    step,
    dtype,
    values,
    indices,
    p_base,
    q_base,
):

    input = true_uniform_quantization_decompress(
        input, quantize_bit, shape, min, step, dtype
    )
    input = input.reshape(-1)
    input[indices] = values
    input = input.reshape(shape)
    error = q_base[0] @ torch.transpose(p_base[0], 1, 2)
    batch, num_head, seq_len, sep_dim = input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    # error = error.permute(0, 2, 1, 3).type(input.dtype)
    error = error.permute(0, 2, 1, 3)
    input = input + error

    return input


def true_uniform_quantization_compress_batchwise(input: torch.Tensor, quantize_bit):

    if quantize_bit != 8 and quantize_bit != 4:
        raise ValueError("quantize_bit should be 8 or 4")
    shape = input.shape
    bsz = shape[0]
    input = input.reshape(bsz, -1)
    if quantize_bit == 8:
        input = input.float()  # convert to 32bits to avoid max - min = inf
    min, max = input.min(dim=-1).values, input.max(dim=-1).values
    step = (max - min) / (pow(2, quantize_bit) - 1)
    min = min.unsqueeze(1)  # Expand min tensor shape to (bsz, 1)
    step = step.unsqueeze(1)  # Expand step tensor shape to (bsz, 1)
    # print("before min max:",min,max,step)
    input = torch.round((input - min) / step)
    # print("after min max:",quantized_input.min(),quantized_input.max())
    # print("quantized isnan:",torch.any(torch.isnan(quantized_input)))
    input = input.to(torch.uint8)
    if quantize_bit == 4:
        input = transfer_8bit_to_4bit_batchwise(input)
    # print("isnan:",torch.any(torch.isnan(returning_input)))
    # while(True):
    #     pass
    return input, shape, min, step


def true_uniform_quantization_decompress_batchwise(
    input: torch.Tensor, quantize_bit, shape, min, step, dtype
):
    if quantize_bit != 8 and quantize_bit != 4:
        raise ValueError("quantize_bit should be 8 or 4")
    bsz = shape[0]
    input = input.reshape(bsz, -1)
    if quantize_bit == 8:
        input = input.float()
        input = input * step + min

        output = input.reshape(shape).type(dtype)
    elif quantize_bit == 4:
        input = transfer_4bit_to_8bit_batchwise(input)

        input = input.type(dtype)
        input = input * step + min
        output = input.reshape(shape)
    return output


def true_outlier_quantization_compress_batchwise(
    input: torch.Tensor, quantize_bit, left
):
    shape = input.shape
    bsz = shape[0]
    input = input.reshape(bsz, -1)
    left_num = int(input.numel() / bsz * left / 2)

    value1, indices1 = torch.topk(input, left_num, largest=False, dim=-1)
    value2, indices2 = torch.topk(input, left_num, largest=True, dim=-1)

    values = torch.cat((value1, value2), dim=-1)
    indices = torch.cat((indices1, indices2), dim=-1)
    # input = input.index_fill_(0,indices,0)
    # print(indices.shape)
    input.scatter_(1, indices, 0)

    output, _, min, step = true_uniform_quantization_compress(input, quantize_bit)

    return output, shape, min, step, values, indices


def true_outlier_quantization_decompress_batchwise(
    input: torch.Tensor, quantize_bit, shape, min, step, dtype, values, indices
):
    bsz = shape[0]
    input = true_uniform_quantization_decompress(
        input, quantize_bit, shape, min, step, dtype
    )
    input = input.reshape(bsz, -1)
    input.scatter_(1, indices, values)
    input = input.reshape(shape)
    return input


def fake_quant_error_simulation_batchwise(input: torch.Tensor, quantize_bit, bsz):
    input = input.reshape(bsz, -1)

    min, max = input.min(dim=-1).values, input.max(dim=-1).values

    step = (max - min) / (pow(2, quantize_bit) - 1)
    min = min.unsqueeze(1)  # Expand min tensor shape to (bsz, 1)
    step = step.unsqueeze(1)  # Expand step tensor shape to (bsz, 1)
    # print("before min max:",min,max,step)
    error = input - torch.round((input - min) / step)
    return error, min, step


def true_gear_compress_batchwise(input: torch.Tensor, quantize_bit, left, rank, loop):
    shape = input.shape
    bsz = shape[0]
    input = input.reshape(bsz, -1)
    left_num = int(input.numel() / bsz * left / 2)
    value1, indices1 = torch.topk(input, left_num, largest=False, dim=-1)
    value2, indices2 = torch.topk(input, left_num, largest=True, dim=-1)
    values = torch.cat((value1, value2), dim=-1)
    indices = torch.cat((indices1, indices2), dim=-1)
    input = input.scatter_(1, indices, 0.0)
    error, min, step = fake_quant_error_simulation_batchwise(input, quantize_bit, bsz)
    error = error.scatter_(1, indices, 0.0)
    error = error.reshape(shape)
    bsz, num_head, seq_len, sep_dim = shape
    smaller_dim = seq_len if seq_len < sep_dim * num_head else sep_dim * num_head
    rank = int(rank * smaller_dim)
    p_base, q_base = true_poweriteration(error, loop, rank)
    # has_inf = torch.isinf(p_base[0])
    # has_nan = torch.isnan(p_base[0])
    # if has_inf.any() or has_nan.any():
    #     print("pbase",has_inf.any(),has_nan.any())
    # has_inf = torch.isinf(q_base[0])
    # has_nan = torch.isnan(q_base[0])
    # if has_inf.any() or has_nan.any():
    #     print("qbase",has_inf.any(),has_nan.any())
    output, _, min, step = true_uniform_quantization_compress(input, quantize_bit)
    return output, shape, min, step, values, indices, p_base, q_base


def true_gear_decompress_batchwise(
    input: torch.Tensor,
    quantize_bit,
    shape,
    min,
    step,
    dtype,
    values,
    indices,
    p_base,
    q_base,
):
    bsz = shape[0]
    input = true_uniform_quantization_decompress(
        input, quantize_bit, shape, min, step, dtype
    )
    input = input.reshape(bsz, -1)
    input.scatter_(1, indices, values)
    input = input.reshape(shape)
    error = q_base[0] @ torch.transpose(p_base[0], 1, 2)
    batch, num_head, seq_len, sep_dim = input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    # error = error.permute(0, 2, 1, 3).type(input.dtype)
    error = error.permute(0, 2, 1, 3)
    input = input + error

    return input


def tokenwise_quantization_compress_with_error(input: torch.Tensor, quantize_bit):
    # Currently only support 4 bit quantization
    assert quantize_bit == 4
    shape = input.shape  # bsz, num_head, seq_len, sep_dim
    input = (
        input.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(shape[0], shape[2], shape[1] * shape[3])
    )
    min, max = input.min(dim=-1).values, input.max(dim=-1).values
    step = (max - min) / (pow(2, quantize_bit) - 1)
    quantized_input = (input - min) / step
    # quantized_input = F.relu(quantized_input)
    quantized_input = quantized_input.round_()
    error = input - quantized_input * step + min
    quantized_input = quantized_input.to(torch.uint8)
    quantized_input = transfer_8bit_to_4bit_batchwise(quantized_input)
    # reshape back to original shape
    # quantized_input = quantized_input.reshape(shape[0],shape[2],shape[1],shape[3])
    error = error.reshape(shape[0], shape[2], shape[1], shape[3])
    # quantized_input = quantized_input.permute(0, 2, 1, 3).contiguous()
    error = error.permute(0, 2, 1, 3).contiguous()
    return quantized_input, error, min, step, shape


def tokenwise_dequantization(quantized_input: torch.Tensor, min, step, shape):
    # input size bsz, seq_len, -1
    quantized_input = transfer_4bit_to_8bit_batchwise(quantized_input)
    quantized_input = quantized_input.to(torch.float16)
    return quantized_input


def true_gear_tokenwiseQ_compress(input: torch.Tensor, quantize_bit, rank, loop):
    shape = input.shape  # bsz, num_head, seq_len, sep_dim
    bsz = shape[0]
    quantized_input, error, min, step, shape = (
        tokenwise_quantization_compress_with_error(input, quantize_bit)
    )
    bsz, num_head, seq_len, sep_dim = shape
    smaller_dim = seq_len if seq_len < sep_dim * num_head else sep_dim * num_head
    rank = int(rank * smaller_dim)
    p_base, q_base = true_poweriteration(error, loop, rank)
    del error
    return quantized_input, shape, min, step, p_base, q_base


def true_gear_tokenwiseQ_decompress(input: torch.Tensor, quantize_bit, rank, loop):
    bsz = shape[0]
    quantized_input, error, min, step, shape = (
        tokenwise_quantization_compress_with_error(input, quantize_bit)
    )
    bsz, num_head, seq_len, sep_dim = shape
    smaller_dim = seq_len if seq_len < sep_dim * num_head else sep_dim * num_head
    rank = int(rank * smaller_dim)
    p_base, q_base = true_poweriteration(error, loop, rank)
    del error
    return quantized_input, shape, min, step, p_base, q_base