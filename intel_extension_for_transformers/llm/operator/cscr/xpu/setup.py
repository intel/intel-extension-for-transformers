from setuptools import setup
import torch
import intel_extension_for_pytorch
from torch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

setup(
    name='gbits_linear',
    ext_modules=[
        DPCPPExtension(
                name='gbits_linear',
                sources=['gbits_linear_new.cpp'],
                extra_compile_args={'cxx': ['-std=c++17', '-fPIC']},
                library_dirs=["/home/sunjiwei/code/intel-extension-for-transformers/intel_extension_for_transformers/llm/operator/cscr/xpu/build/"],
                libraries=["xetla_linear"],
                ),
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension
    },
    )
