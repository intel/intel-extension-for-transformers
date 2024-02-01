#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

WORLD_AVG_CARBON_INTENSITY = 0.475   # kgCO2e/kWh
POWER_PER_GB_MEM = 0.1      # roughly ratio in DDR5
JOUL_TO_KWH = 2.78e-7

def main():
    parser = argparse.ArgumentParser(description='LLM carbon calculator - '
                                    'simple estimator of LLM inference '
                                    'carbon emission')
    parser.add_argument('-c', '--carbon-intensity', type=float,
                        dest='carbon_intensity',
                        default=WORLD_AVG_CARBON_INTENSITY, metavar='C',
                        help='carbon intensity of electricity of your country '
                        'or cloud provider (default: 0.475 - world average)')
    parser.add_argument('-t', '--time', type=float, metavar='T',
                        help='total time of one inference procedure'
                        'in mini-seconds')
    parser.add_argument('--fl', '--first-latency', type=float, metavar='FTL',
                        dest='first_latency',
                        help='first token latency in mini-seconds')
    parser.add_argument('--nl', '--next-latency', type=float, metavar='NTL',
                        dest='next_latency',
                        help='next token latency in mini-seconds')
    parser.add_argument('-n', '--token-size', default=32, type=int, metavar='N',
                        dest='token_size',
                        help='output token number in one inference (default: 32)')
    parser.add_argument('--tdp', type=int, required=True, metavar='TDP',
                        help='device TDP in Watts, it could be '
                        'CPU/GPU/Accelerators')
    parser.add_argument('-m', '--mem', type=float, metavar='M', required=True,
                        help='memory consumption in MB')


    args = parser.parse_args()

    if args.time is not None:
        t = args.time
    elif args.first_latency is not None and args.next_latency is not None:
        t = args.first_latency + args.next_latency * (args.token_size - 1)
    else:
        print('You need to specify either total time of '
            'one inference (-t) or both first token latency (--fl) '
            'and next token latency(--nl)')
        return 0.0

    m = args.mem
    tdp = args.tdp
    c = args.carbon_intensity
    carbon = (tdp + m * 0.001 * POWER_PER_GB_MEM) * t * 0.001 * JOUL_TO_KWH * c

    print('TDP (W): ', tdp)
    print('Memory Consumption (MB): ', m)
    print('Output token number: ', args.token_size)
    print('Total time of one inference (ms): ', t)
    print('Carbon emission in one inference (kgCO2e): ', carbon)
    return carbon

if __name__ == '__main__':
    main()
