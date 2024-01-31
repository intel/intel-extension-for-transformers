# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

import argparse

from itrex_opt import ItrexOpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--local-rank", type=int, required=False)
    parser.add_argument("--local_rank", type=int, required=False)
    parser.add_argument("--no_cuda", action="store_true", required=False, default=False)
    args = parser.parse_args()

    itrex_opt = ItrexOpt(args.config_file, args.no_cuda)
    itrex_opt.e2e()


if __name__ == "__main__":
    main()
