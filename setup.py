from io import open
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
import os
import re
import sys
import shutil
from packaging.version import Version
from subprocess import check_output

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    filepath = './intel_extension_for_transformers/version.py'
    with open(filepath) as version_file:
        __version__, = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    if path == None:
      return None
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == 'win32':
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            if exts == None:
              return None
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


def get_version(cmd):
    "Returns cmake version."
    try:
        for line in check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                print(line.strip().split(' ')[2])
                return Version(line.strip().split(' ')[2])
    except Exception as error:
        return Version('0.0.0')


def get_cmake_command():
    "Returns cmake command."

    cmake_command = 'cmake'
    if sys.platform == 'win32':
        return cmake_command
    cmake3 = which('cmake3')
    cmake = which('cmake')
    if cmake3 is not None:
        if cmake is not None:
            bare_version = get_version('cmake')
            if (bare_version < Version("3.12.0") and get_version('cmake3') > bare_version):
                cmake_command = 'cmake3'
        else:
            cmake_command = 'cmake3'
    elif cmake is None:
        raise RuntimeError('no cmake or cmake3 found')
    return cmake_command


class build_ext(build_ext):
    def build_extension(self, ext):
        if not sys.platform.startswith("win"):
            import pathlib
            cwd = pathlib.Path().absolute()

            build_temp = pathlib.Path(self.build_temp)
            build_temp.mkdir(parents=True, exist_ok=True)
            extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
            executable_path = extdir.parent.absolute()
            cmake_args = [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
                '-DPYTHON_EXECUTABLE={}'.format(sys.executable)
            ]

            build_args = ['-j']
            cmake_command = get_cmake_command()
            os.chdir(str(build_temp))
            self.spawn([cmake_command, ext.sourcedir] + cmake_args)
            self.spawn(['make'] + build_args)
            if os.path.exists('neural_engine'):
                shutil.copy('neural_engine', executable_path)
            os.chdir(str(cwd))
        else:
            print("Neural Engine is not support windows for now")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def check_submodules():
    def check_for_files(folder, files):
        if not any(os.path.exists(os.path.join(folder, f)) for f in files):
            report("Could not find any of {} in {}".format(", ".join(files), folder))
            report("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    def not_exists_or_empty(folder):
        return not os.path.exists(folder) or (os.path.isdir(folder)
                                              and len(os.listdir(folder)) == 0)

    git_modules_path = os.path.join(cwd, ".gitmodules")
    with open(git_modules_path) as f:
        folders = [
            os.path.join(cwd,
                         line.split("=", 1)[1].strip()) for line in f.readlines()
            if line.strip().startswith("path")
        ]

    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder)
           for folder in folders) and not sys.platform.startswith("win"):
        try:
            print(' --- Trying to initialize submodules')
            start = time.time()
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=cwd)
            end = time.time()
            print(' --- Submodule initialization took {:.2f} sec'.format(end - start))
        except Exception:
            print(' --- Submodule initalization failed')
            print('Please run:\n\tgit submodule update --init --recursive')
            sys.exit(1)


if __name__ == '__main__':
    check_submodules()

    setup(
        name="intel_extension_for_transformers",
        version=__version__,
        author="Intel AIA/AIPC Team",
        author_email=
        "feng.tian@intel.com, haihao.shen@intel.com,hanwen.chang@intel.com, penghui.cheng@intel.com",
        description="Repository of Intel® Extension for Transformers",
        long_description=open("README.md", "r", encoding='utf-8').read(),
        long_description_content_type="text/markdown",
        keywords=
        'quantization, auto-tuning, post-training static quantization, post-training dynamic quantization, quantization-aware training, tuning strategy',
        license='Apache 2.0',
        url="https://github.com/intel/",
        ext_modules=[
            CMakeExtension("neural_engine_py",
                           str(cwd) + '/intel_extension_for_transformers/backends/neural_engine/executor/')
        ],
        packages=find_packages(),
        include_package_data=True,
        package_dir={'': '.'},
        package_data={
            '': ['*.py', '*.yaml'],
            'executor': ['*.py'],
        },
        cmdclass={
            'build_ext': build_ext,
        },
        install_requires=['numpy', 'transformers>=4.12.0', 'packaging'],
        scripts=['intel_extension_for_transformers/backends/neural_engine/bin/neural_engine'],
        python_requires='>=3.7.0',
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: Apache Software License',
        ],
    )
