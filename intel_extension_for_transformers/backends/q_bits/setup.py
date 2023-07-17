import glob
import os
import sys
import subprocess
from cmake import CMAKE_BIN_DIR
from pathlib import Path
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# define install requirements
# install_requires_list = ['neural_compressor']
# libs = list(glob.glob("./bitsandbytes/libbitsandbytes*.so"))
# libs = [os.path.basename(p) for p in libs]
# print("libs:", libs)

cwd = os.path.dirname(os.path.abspath(__file__))


CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE", "Release")
""" Whether to build with -O0 / -O3 / -g; could be one of Debug / Release / RelWithDebInfo; default to Release """

CMAKE_GENERATOR = os.environ.get("CMAKE_GENERATOR", "Ninja")
""" The CMake generator to be used; default to Ninja """

CMAKE_ARGS = os.environ.get("CMAKE_ARGS", "")
""" Adding CMake arguments set as environment variable (needed e.g. to build for GPU support on conda-forge) """

CMAKE_BUILD_PARALLEL_LEVEL = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "")
""" Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level across all generators """


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


class CMakeExtension(Extension):
    """CMakeExtension class."""

    def __init__(self, name, sourcedir=""):
        """Init a CMakeExtension object."""
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Extension builder."""

    _copy_targes: bool = False

    @staticmethod
    def _is_target_file(file_name: str) -> bool:
        if file_name.endswith(".dll") or file_name.endswith(".exe"):
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
            f"-DQBITS_VERSION_STRING={self.distribution.get_version()}",
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


setup(
    name=f"q_bits",
    version=f"0.0",
    author="Intel AIA/AIPC Team",
    author_email="feng.tian@intel.com, haihao.shen@intel.com,hanwen.chang@intel.com, penghui.cheng@intel.com",
    description="k-bit optimizers and matrix multiplication routines.",
    license="Apache 2.0",
    keywords="quantization, weight_only, 8-bits, 4-bits",
    url="https://github.com/intel/intel-extension-for-transformers/backend/q_bits",
    packages=find_packages(),
    # package_data={"": libs},
    ext_modules=[CMakeExtension(
            "q_bits.q_bits_py", 'cscr')],
    cmdclass={'build_ext': CMakeBuild},
    entry_points={
        'console_scripts': [
            'q_bits = q_bits:q_bits_bin',
        ]
    },
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
    ],
)