"""Setup and install modules."""
import importlib
import os
import subprocess
import sys
import time
from io import open
from pathlib import Path
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def get_gpu_family():
    ''' Get gpu device family info.

    Return 'flex'|'max'|'arc'| 'no_gpu'| assert

    Note, this function need to import intel_extension_for_pytorch

    Additional info (common gpu name):
      'Intel(R) Data Center GPU Flex 170'
      'Intel(R) Data Center GPU Max 1100'
      'Intel(R) Arc(TM) A770 Graphics'
    '''

    import torch
    import intel_extension_for_pytorch as ipex
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return 'no_gpu'

    name = torch.xpu.get_device_name()
    if 'GPU Flex' in name:
        return 'flex'
    if 'GPU Max' in name:
        return 'max'
    if 'Arc(TM)' in name:
        return 'arc'
    assert False, "Unsupported GPU device: {}".format(name)


def check_env_flag(name: str, default: bool = False) -> bool:
    if default:  # if a flag meant to be true if not set / mal-formatted
        return not os.getenv(name, "").upper() in ["OFF", "0", "FALSE", "NO", "N"]
    else:
        return os.getenv(name, "").upper() in ["ON", "1", "TRUE", "YES", "Y"]


SKIP_RUNTIME = check_env_flag("SKIP_RUNTIME", False)
""" Whether to only packaging optimization """

RUNTIME_ONLY = check_env_flag("RUNTIME_ONLY", False)
""" Whether to only packaging backends """

ipex_available = importlib.util.find_spec("intel_extension_for_pytorch") is not None
IS_INTEL_GPU = False
if ipex_available and (get_gpu_family() != "no_gpu"):
    SKIP_RUNTIME = True
    RUNTIME_ONLY = False
    IS_INTEL_GPU = True

if not SKIP_RUNTIME:
    from cmake import CMAKE_BIN_DIR
    from cpuinfo import get_cpu_info
    cpu_flags = get_cpu_info()['flags']


    CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE", "Release")
    """ Whether to build with -O0 / -O3 / -g; could be one of Debug / Release / RelWithDebInfo; default to Release """

    CMAKE_GENERATOR = os.environ.get("CMAKE_GENERATOR", "Ninja")
    """ The CMake generator to be used; default to Ninja """

    CMAKE_ARGS = os.environ.get("CMAKE_ARGS", "")
    """ Adding CMake arguments set as environment variable (needed e.g. to build for GPU support on conda-forge) """

    CMAKE_BUILD_PARALLEL_LEVEL = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "")
    """ Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level across all generators """

    NE_WITH_AVX2 = check_env_flag("NE_WITH_AVX2", 'avx512f' not in cpu_flags)
    """ Whether to limit the max ISA used to AVX2; otherwise AVX512 will be used; set to ON/OFF """

cwd = os.path.dirname(os.path.abspath(__file__))

# define install requirements
install_requires_list = ['packaging', 'numpy', 'schema', 'pyyaml']
opt_install_requires_list = ['neural_compressor', 'transformers']


packages_list = find_packages()
install_requires_list.extend(opt_install_requires_list)


class CMakeExtension(Extension):
    """CMakeExtension class."""

    def __init__(self, name, sourcedir="", lib_only=False):
        """Init a CMakeExtension object."""
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.optional = lib_only  # we only deliver shared object but not as a python extension module


class CMakeBuild(build_ext):
    """Extension builder."""

    _copy_targes: bool = False

    @staticmethod
    def _is_target_file(file_name: str) -> bool:
        if file_name.endswith(".dll") or file_name.endswith(".exe") or file_name.endswith(".pyd"):
            return True
        if file_name.endswith(".so") or ".so." in file_name:
            return True
        if sys.platform == "linux" and ('.' not in file_name):
            return True
        return False

    @staticmethod
    def _get_files(scope: str, repo: str):
        ''' Equivalent of `git ls-files --recurse-submodules -- $scope` for git-v1.x '''
        files = [os.path.join(repo, f) for f in subprocess.check_output(
                ["git", "ls-files", "--", scope], cwd=repo
        ).decode("utf-8").splitlines()]
        submodules = subprocess.check_output(
            ["git", "submodule", "--quiet", "foreach", f'echo $sm_path'], cwd=repo).decode("utf-8").splitlines()
        for sm in submodules:
            sm_path = os.path.join(repo, sm)
            files.extend(CMakeBuild._get_files(sm_path, sm_path))
        return files

    def get_source_files(self):
        """ The primary purpose of this function is to help populating the `sdist` with all the files necessary to build the distribution. -- setuptools doc"""
        files = super().get_source_files()
        if not os.path.isdir(os.path.join(cwd, ".git")):
            return files

        for ext in self.extensions:
            if not isinstance(ext, CMakeExtension):
                continue
            files.extend(os.path.relpath(f, cwd)
                         for f in self._get_files(ext.sourcedir, cwd))
        return files

    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        output_dir = f"{extdir}{os.sep}"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{CMAKE_BUILD_TYPE.upper()}={output_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{CMAKE_BUILD_TYPE.upper()}={output_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={CMAKE_BUILD_TYPE}",
            f"-DNE_VERSION_STRING={self.distribution.get_version()}",
            f"-DNE_WITH_AVX2={'ON' if NE_WITH_AVX2 else 'OFF'}",
            f"-DNE_WITH_TESTS=OFF",
        ]
        if sys.platform == "linux":  # relative_rpath
            cmake_args.append('-DCMAKE_BUILD_RPATH=$ORIGIN/')

        build_args = []
        my_env: dict[str, str] = os.environ.copy()

        # Using Ninja-build since it a) is available as a wheel and b)
        # multithreads automatically. MSVC would require all variables be
        # exported for Ninja to pick it up, which is a little tricky to do.
        generator = CMAKE_GENERATOR

        if generator == "Ninja":
            try:
                import ninja

                ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                cmake_args += [
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                ]
            except ImportError:
                generator = ""
        if generator:
            cmake_args += [f"-G{generator}"]

        if self.compiler.compiler_type == "msvc":

            # Single config generators are handled "normally"
            single_config = any(x in generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in generator for x in {"Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            PLAT_TO_CMAKE = {  # Convert distutils Windows platform specifiers to CMake -A arguments
                "win32": "Win32",
                "win-amd64": "x64",
            }
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            if generator == "Ninja":
                # temporary solution based on that of pytorch
                from distutils import _msvccompiler  # type: ignore[import]
                vc_env = _msvccompiler._get_vc_env("x64")
                # Keys in `_get_vc_env` are always lowercase while OS environ keys are always uppercase on Windows.
                # https://stackoverflow.com/a/7797329
                my_env = {**my_env, **{k.upper(): v for k, v in vc_env.items()}}

            # Multi-config generators have a different way to specify configs
            if not single_config:
                build_args += ["--config", CMAKE_BUILD_TYPE]

        if CMAKE_ARGS:
            cmake_args += [item for item in CMAKE_ARGS.split(" ") if item]

        if not CMAKE_BUILD_PARALLEL_LEVEL:
            parallel_level = getattr(self, 'parallel', '') or ''
            build_args += [f"-j{parallel_level}"]

        # we avoid using self.build_tmp for incremental builds
        build_dir = Path("build") / ext.name.split('.')[-1]
        if not build_dir.exists():
            build_dir.mkdir(parents=True)
        cmake_path = os.path.join(CMAKE_BIN_DIR, "cmake")
        config_command = [cmake_path, *cmake_args, ext.sourcedir]
        build_command = [cmake_path, "--build", ".", *build_args]
        print(' '.join(config_command))
        subprocess.run(config_command, cwd=build_dir, check=True, env=my_env)
        print(' '.join(build_command))
        subprocess.run(build_command, cwd=build_dir, check=True, env=my_env)
        if (self._copy_targes):
            for f in next(os.walk(output_dir))[2]:
                if CMakeBuild._is_target_file(f):
                    self.copy_file(
                        os.path.join(output_dir, f),
                        os.path.join(cwd, *ext.name.split('.')[:-1], f)
                    )

    def get_output_mapping(self):
        mapping: dict[str, str] = getattr(super(), 'get_output_mapping')()
        for ext in self.extensions:
            if not isinstance(ext, CMakeExtension):
                continue
            build_lib = (Path(self.build_lib) /
                         ext.name.replace('.', os.sep)).parent
            ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
            for f in next(os.walk(build_lib.resolve()))[2]:
                mapping[str(build_lib / f)] = str(ext_dir / f)
        return mapping

    def run(self) -> None:
        self._copy_targes = self.inplace or \
            getattr(self, 'editable_mode', False)
        return super().run()


def check_submodules():
    """Check submodules information."""
    if not os.path.exists(".git"):
        return
    try:
        print(' --- Trying to initialize submodules')
        start = time.time()
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"], cwd=cwd)
        end = time.time()
        print(f' --- Submodule initialization took {end - start:.2f} sec')
    except Exception:
        print(' --- Submodule initialization failed')
        print('Please run:\n\tgit submodule update --init --recursive')
        sys.exit(1)


if __name__ == '__main__':
    if IS_INTEL_GPU:
        ext_modules = []
    else:
        ext_modules = [CMakeExtension(
            "intel_extension_for_transformers.qbits", 'intel_extension_for_transformers/llm/operator/csrc', lib_only=True)]
    if not SKIP_RUNTIME:
        check_submodules()
        ext_modules.extend([
            CMakeExtension("intel_extension_for_transformers.neural_engine_py", "intel_extension_for_transformers/llm/runtime/deprecated/"),
            ])
    cmdclass={'build_ext': CMakeBuild}

    setup(
        name="intel-extension-for-transformers",
        author="Intel AIA/AIPC Team",
        author_email="feng.tian@intel.com, haihao.shen@intel.com,hanwen.chang@intel.com, penghui.cheng@intel.com",
        description="Repository of Intel® Intel Extension for Transformers",
        long_description=open("README.md", "r", encoding='utf-8').read(),
        long_description_content_type="text/markdown",
        keywords='quantization, auto-tuning, post-training static quantization, post-training dynamic quantization, quantization-aware training, tuning strategy',
        license='Apache 2.0',
        url="https://github.com/intel/intel-extension-for-transformers",
        ext_modules=ext_modules,
        packages=find_packages(),
        package_dir={'': '.'},
        # otherwise CMakeExtension's source files will be included in final installation
        include_package_data=False,
        package_data={
            '': ["*.yaml", "*.mat"],
        },
        cmdclass=cmdclass if not SKIP_RUNTIME else {},
        install_requires=install_requires_list,
        entry_points={
            'console_scripts': [
                'neural_engine = intel_extension_for_transformers.llm.runtime.deprecated:neural_engine_bin',
                'neuralchat = intel_extension_for_transformers.neural_chat.cli.cli_commands:neuralchat_execute',
                'neuralchat_server = intel_extension_for_transformers.neural_chat.server.server_commands:neuralchat_server_execute',
                'neuralchat_client = intel_extension_for_transformers.neural_chat.server.server_commands:neuralchat_client_execute'
            ]
        },
        python_requires='>=3.7.0',
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: Apache Software License',
        ],
        setup_requires=['setuptools_scm'],
        use_scm_version=True,
    )
