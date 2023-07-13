#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import cmd
import os
import platform
import sys
import glob
import argparse
import fnmatch
import subprocess

ProjectEXT=['h','hpp','c','cpp']

def glob_files(dirs):
    files = []
    for directory in dirs:
        for root, _, filenames in os.walk(directory):
            for ext in ProjectEXT:
                for filename in fnmatch.filter(filenames, '*.' + ext):
                    files.append(os.path.join(root, filename))
    return files

if sys.platform == "linux":
    ClangBin='clang-format'
elif sys.platform == 'win32':
    ClangBin='clang-format.exe'

def clang_format_dir(args):
    files=glob_files(args.dirs)
    for file in files:
        cmds=[ClangBin,'-i','--style=file',file]
        subprocess.run(cmds, check=True)

def parse_args(argv=None):
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(description='Recursively clang-format')
    parser.add_argument('--dirs',nargs='+',
                        help='paths to clang-format')
    args = parser.parse_args(argv[1:])
    if not args.dirs:
        sys.exit(-1)
    return args

if __name__=='__main__':
    if len(sys.argv)==1:
        args=parse_args(['','--dirs','core','jblas','models','vectors','application'])
    else:
        args=parse_args()
    clang_format_dir(args)
