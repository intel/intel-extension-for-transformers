
import torch
from torch import nn


class ParamsForBits(torch.nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, quant_state=None, blocksize=32, compress_statistics=True, quant_type='int8'):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.data = data
        return self


class QuantizedLinearBits(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype="int8",
        compress_statistics=True,
        quant_bits=8,
        quant_type='int8',
        blocksize=32,
        scheme="sym",
        device=None,
    ):
        super().__init__(input_features, output_features, bias, device)
        self.compute_dtype = compute_dtype
        self.compress_statistics = compress_statistics
        self.blocksize = blocksize
        self.scheme = scheme
        self.quant_bits = quant_bits
        self.quant_type = quant_type

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if getattr(self.weight, 'quant_state', None) is None:
            print('FP4 quantization state not initialized. Please call .quantize_weights().')

        m = x.size()[0]
        out = torch.zeros(m, self.out_features, dtype=torch.float)
        torch.ops.weight_only_jblasop.jblas_quantweight_f32_linear(
            x, self.weight.data, out, m, self.out_features, self.in_features, self.in_features, self.out_features)

        return out
    
    def set_weights(self, data):
        weight = torch.ops.weight_only_jblasop.jblas_quantize(
            data, True, self.quant_bits, self.scheme, self.blocksize, self.compute_dtype)
        quant_type = self.quant_type
        self.weight = ParamsForBits(
            data=weight, requires_grad=False, quant_state={"scheme": self.scheme}, blocksize=self.blocksize,
            compress_statistics=self.compress_statistics, quant_type=quant_type
        )


class QuantizedLinearINT4(QuantizedLinearBits):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype="fp32",
        compress_statistics=True,
        blocksize=32,
        scheme="sym",
        device=None,
    ):
        super().__init__(input_features, output_features, bias, compute_dtype, compress_statistics,
                         4, "int4", blocksize, scheme, device)

class QuantizedLinearINT8(QuantizedLinearBits):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype="fp32",
        compress_statistics=True,
        blocksize=32,
        scheme="sym",
        device=None,
    ):
        super().__init__(input_features, output_features, bias, compute_dtype, compress_statistics,
                         8, "int8", blocksize, scheme, device)