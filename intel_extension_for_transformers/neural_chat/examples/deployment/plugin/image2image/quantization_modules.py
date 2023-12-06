import torch
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize, FakeQuantizeBase, default_fake_quant, default_per_channel_weight_fake_quant, default_fused_act_fake_quant, default_fused_per_channel_wt_fake_quant
from torch.ao.nn.quantized import Linear, Conv2d, Quantize
try:
    from peft.tuners.lora import Linear as LoRALinear
    from peft.utils import transpose
    peft_available = True
except:
    peft_available = False

ACT_REDUCE_RANGE = False
WEIGHT_REDUCE_RANGE = False


class QuantizedLinear(Linear):
    def forward(self, x):
        return super().forward(
            self.input_quant(x)
        ).dequantize()


class FakeQuantLinear(torch.nn.Linear):
    def __init__(self, module: torch.nn.Linear):
        self.__dict__.update(module.__dict__.copy())
        self.add_module('activation_pre_process', default_fake_quant(reduce_range=ACT_REDUCE_RANGE))
        self.add_module('weight_fake_quant', default_per_channel_weight_fake_quant(reduce_range=WEIGHT_REDUCE_RANGE))
        self.add_module('activation_post_process', default_fake_quant(reduce_range=ACT_REDUCE_RANGE))
        self.is_lora_layer = True if peft_available and isinstance(module, LoRALinear) else False

    def forward(self, x):
        x = self.activation_pre_process(x)
        weight = self.weight
        if self.is_lora_layer and not self.disable_adapters and self.r[self.active_adapter] > 0:
            lora_weight = transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                ) * self.scaling[self.active_adapter]
            weight = weight + lora_weight
        x = F.linear(x, self.weight_fake_quant(weight), self.bias)
        x = self.activation_post_process(x)
        return x

    def convert(self):
        if self.is_lora_layer and not self.disable_adapters and self.r[self.active_adapter] > 0:
            lora_weight = transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                ) * self.scaling[self.active_adapter]
            self.weight.data += lora_weight.data
        module = QuantizedLinear.from_float(self)
        input_quant = torch.quantization.QuantStub()
        input_quant.add_module('activation_post_process', self.activation_pre_process)
        input_quant = Quantize.from_float(input_quant)
        module.add_module('input_quant', input_quant)
        return module


class QuantizedConv2d(Conv2d):
    def forward(self, x):
        return super().forward(
            self.input_quant(x)
        ).dequantize()


class FakeQuantConv2d(torch.nn.Conv2d):
    def __init__(self, module: torch.nn.Conv2d):
        self.__dict__.update(module.__dict__.copy())
        self.add_module('activation_pre_process', default_fake_quant(reduce_range=ACT_REDUCE_RANGE))
        self.add_module('weight_fake_quant', default_per_channel_weight_fake_quant(reduce_range=WEIGHT_REDUCE_RANGE))
        self.add_module('activation_post_process', default_fake_quant(reduce_range=ACT_REDUCE_RANGE))

    def forward(self, x):
        x = self.activation_pre_process(x)
        x = self._conv_forward(x, self.weight_fake_quant(self.weight), self.bias)
        x = self.activation_post_process(x)
        return x

    def convert(self):
        module = QuantizedConv2d.from_float(self)
        input_quant = torch.quantization.QuantStub()
        input_quant.add_module('activation_post_process', self.activation_pre_process)
        input_quant = Quantize.from_float(input_quant)
        module.add_module('input_quant', input_quant)
        return module


def get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def find_and_replace(model, fake_quant=True):
    assert isinstance(model, torch.nn.Module), "Only support torch Module."
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        try:
            parent, target, target_name = get_submodules(model, key)
        except:
            continue
        if fake_quant:
            if isinstance(target, torch.nn.Linear):
                setattr(parent, target_name, FakeQuantLinear(target))
            elif isinstance(target, torch.nn.Conv2d):
                setattr(parent, target_name, FakeQuantConv2d(target))
        else:
            if isinstance(target, FakeQuantLinear):
                setattr(parent, target_name, target.convert())
            elif isinstance(target, FakeQuantConv2d):
                setattr(parent, target_name, target.convert())

def convert2quantized_model(model):
    model.to(torch.device("cpu"))
    find_and_replace(model, fake_quant=False)
    return model

def disable_all_observers(model):
    assert isinstance(model, torch.nn.Module), "Only support torch Module."
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizeBase):
            module.disable_observer()

def sync_all_observers(model):
    assert isinstance(model, torch.nn.Module), "Only support torch Module."
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantize):
            _scale, _zero_point = module.calculate_qparams()
            _scale, _zero_point = _scale.to(module.scale.device), _zero_point.to(module.zero_point.device)
            if module.scale.shape != _scale.shape:
                module.scale.resize_(_scale.shape)
                module.zero_point.resize_(_zero_point.shape)
            module.scale.copy_(_scale)
            module.zero_point.copy_(_zero_point)

def load_int8_model(fp32_model, int8_model_path, fake_quantize_model=False):
    find_and_replace(fp32_model)
    if fake_quantize_model:
        fp32_model.load_state_dict(torch.load(int8_model_path))
        disable_all_observers(fp32_model)
        sync_all_observers(fp32_model)
    int8_model = convert2quantized_model(fp32_model)
    print('Converted to quantized model.')
    if not fake_quantize_model:
        int8_model.load_state_dict(torch.load(int8_model_path))
    return int8_model