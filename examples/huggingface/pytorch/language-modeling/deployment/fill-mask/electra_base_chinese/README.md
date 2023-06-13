Step-by-Step
=========
This document describes the end-to-end workflow for Huggingface model [Electra Base Chinese Generator](https://huggingface.co/hfl/chinese-legal-electra-base-generator) and [Electra Base Chinese Discriminator](https://huggingface.co/hfl/chinese-electra-base-discriminator) of fill_mask task with Neural Engine backend.

# Prerequisite
## Prepare Python Environment
```shell
# Create Environment (conda)
conda create -n <env name> python=3.8
conda activate <env name>
```

Check the gcc version using `gcc -v`, make sure the `gcc` version is higher than 9.0.
If not, you need to update `gcc` by yourself.

Install Intel® Extension for Transformers, please refer to [installation](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/installation.md)
```shell
# Install from pypi
pip install intel-extension-for-transformers

# Install from source code
cd <intel_extension_for_transformers_folder>
pip install -v .
```

Install required dependencies for examples
```shell
cd <intel_extension_for_transformers_folder>/examples/huggingface/pytorch/language-modeling/deployment/fill-mask/electra_base_chinese
pip install -r requirements.txt
```

## Environment Variables (Optional)
```shell
# Preload libjemalloc.so may improve the performance when inference under multi instance.
conda install jemalloc==5.2.1 -c conda-forge -y
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so

# Using weight sharing can save memory and may improve the performance when multi instances.
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
>**Note**: This step is optional.

# Inference Pipeline

Neural Engine can parse PyTorch JIT Traced model and generate Neural Engine IR. In this example, we provide the FP32 and BF16 inference pipelines.

## Generate IR
```shell
# fp32 generator
python export_ir.py --model_name=hfl/chinese-legal-electra-base-generator --dtype=fp32 --output_model=gen_fp32_ir --pt_file=gen_model.pt

# bf16 generator
python export_ir.py --model_name=hfl/chinese-legal-electra-base-generator --dtype=bf16 --output_model=gen_bf16_ir --pt_file=gen_model.pt

# fp32 discriminator
python export_ir.py --model_name=hfl/chinese-electra-base-discriminator --dtype=fp32 --output_model=dis_fp32_ir --pt_file=disc_model.pt

# bf16 discriminator
python export_ir.py --model_name=hfl/chinese-electra-base-discriminator --dtype=bf16 --output_model=dis_bf16_ir --pt_file=disc_model.pt
```

| arg | meaning | default |
| :--: | :-----: | :-----: |
| model_name | huggingface pretrained model name or path in your local machine | hfl/chinese-legal-electra-base-generator |
| dtype | FP32 or BF16 | FP32 |
| output_model | path where the generated IR file will be stored in your local machine | ./ir |
| pt_file | path to the pytorch jit traced model file in your local machine | ./model.pt |

If you supply the related jit trace pt model, it will not perform trace process during IR generation.

## Try Fill_Mask Task
### Generator
```shell
# fp32
python run_electra.py --generator_ir_path=gen_fp32_ir --generator_model_name=hfl/chinese-legal-electra-base-generator --text=其实了解一个人并不代[MASK]什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的 --generator_or_discriminator=generator

# bf16
python run_electra.py --generator_ir_path=gen_bf16_ir --generator_model_name=hfl/chinese-legal-electra-base-generator --text=其实了解一个人并不代[MASK]什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的 --generator_or_discriminator=generator
```

The terminal output like this:
```shell
Complete text given by PyTorch model:
'>>> 其实了解一个人并不代表什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
'>>> 其实了解一个人并不代管什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
'>>> 其实了解一个人并不代说什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
==========================================================================
Complete text given by Neural Engine model:
'>>> 其实了解一个人并不代表什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
'>>> 其实了解一个人并不代管什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
'>>> 其实了解一个人并不代说什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
==========================================================================
```

You can try it by sending another Chinese sentence with a tag `[MASK]`.

### Discriminator
```shell
# fp32
python run_electra.py --discriminator_ir_path=dis_fp32_ir --discriminator_model_name=hfl/chinese-electra-base-discriminator --text=其实了解一个人并不代表什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的 --generator_or_discriminator=discriminator

# bf16
python run_electra.py --discriminator_ir_path=dis_bf16_ir --discriminator_model_name=hfl/chinese-electra-base-discriminator --text=其实了解一个人并不代表什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的 --generator_or_discriminator=discriminator
```

The terminal output like this:
```shell
Discriminator input text token: ['[CLS]', '其', '实', '了', '解', '一', '个', '人', '并', '不', '代', '表', '什', '么', '，', '人', '是', '会', '变', '的', '，', '今', '天', '他', '喜', '欢', '凤', '梨', '，', '明', '天', '他', '可', '以', '喜', '欢', '别', '的', '[SEP]']
Discrimination labels given by PyTorch model:
'>>> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
==========================================================================
Discrimination labels given by Neural Engine model:
'>>> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
==========================================================================
```

You can try it by sending another Chinese sentence without a tag `[MASK]`.

You can also use generator and discriminator together. In this case, the discriminator will use the text generated by the generator.
```shell
# fp32
python run_electra.py --generator_ir_path=gen_fp32_ir --generator_model_name=hfl/chinese-legal-electra-base-generator --discriminator_ir_path=dis_fp32_ir --discriminator_model_name=hfl/chinese-electra-base-discriminator --text=其实了解一个人并不代[MASK]什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的

# bf16
python run_electra.py --generator_ir_path=gen_bf16_ir --generator_model_name=hfl/chinese-legal-electra-base-generator --discriminator_ir_path=dis_bf16_ir --discriminator_model_name=hfl/chinese-electra-base-discriminator --text=其实了解一个人并不代[MASK]什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的
```

The terminal output like this:
```shell
Complete text given by PyTorch model (Top-3):
'>>> 其实了解一个人并不代表什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
'>>> 其实了解一个人并不代管什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
'>>> 其实了解一个人并不代说什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
==========================================================================
Complete text given by Neural Engine model (Top-3):
'>>> 其实了解一个人并不代表什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
'>>> 其实了解一个人并不代管什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
'>>> 其实了解一个人并不代说什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的'
==========================================================================
Discriminator input text token: ['[CLS]', '其', '实', '了', '解', '一', '个', '人', '并', '不', '代', '表', '什', '么', '，', '人', '是', '会', '变', '的', '，', '今', '天', '他', '喜', '欢', '凤', '梨', '，', '明', '天', '他', '可', '以', '喜', '欢', '别', '的', '[SEP]']
Discrimination labels given by PyTorch model:
'>>> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
==========================================================================
Discrimination labels given by Neural Engine model:
'>>> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
==========================================================================
```

| arg | meaning | default |
| :--: | :-----: | :-----: |
| generator_ir_path | path to generator fp32 or bf16 IR files | ./ir |
| discriminator_ir_path | path to gdiscriminator fp32 or bf16 IR files | ./ir |
| generator_model_name | huggingface pretrained model name or path in your local machine | hfl/chinese-legal-electra-base-generator |
| discriminator_model_name | huggingface pretrained model name or path in your local machine | hfl/chinese-electra-base-discriminator |
| generator_or_discriminator | choose model, choices=[generator, discriminator, both] | both |
| text | a Chinese sentence with a tag '[MASK]' when using generator or without a tag '[MASK]' when only using discriminator | 其实了解一个人并不代[MASK]什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的 |

## Performance
You can also test the IR performace with different batch_size and sequence length.
```shell
# use numactl to bind cores for better performance
# support single socket and multiple sockets
numactl -m <node N> -C <cpu list> python run_electra.py --generator_ir_path=<gen_fp32_ir / gen_bf16_ir> --discriminator_ir_path=<dis_fp32_ir / dis_bf16_ir> --batch_size=4 --seq_len=128 --mode=performance
```
| arg | meaning | default |
| :--: | :-----: | :-----: |
| generator_ir_path | path to generator fp32 or bf16 IR files | ./ir |
| discriminator_ir_path | path to gdiscriminator fp32 or bf16 IR files | ./ir |
| generator_or_discriminator | choose model, choices=[generator, discriminator, both] | both |
| mode | test ir performance or not, choices=[accuracy, performance]| accuracy |
| batch_size | input batch size | 1 |
| seq_len | input sequence length | 128 |
| iterations | iteration in performance mode | 10 |
| warm_up | warm up iteration in performance mode | 5 |
| log_file | file path to log information | executor.log |
