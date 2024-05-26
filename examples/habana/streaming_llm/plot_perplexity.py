# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapted from https://github.com/tomaarsen/attention_sinks

First run `run_streaming_llm.py` to generate one or more `csv` files.
This script can plot those csv files.

Usage:
python benchmark/plot_perplexity.py
python benchmark/plot_perplexity.py --features perplexity latency --title "Log perplexity & latency of Llama 2 7B as a function of input lengths"
"""


import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

FEATURE_DF_MAP = {
    "perplexity": "overall_ppl",
    "memory": "hpu_ram_allocated",
    "latency": "latency",
}
FEATURE_STYLE_MAP = {
    "perplexity": "-",
    "memory": "--",
    "latency": ":",
}
FEATURE_LABEL_MAP = {
    "perplexity": "Perplexity (log), lower is better",
    "memory": "HPU RAM Usage (GB), lower is better",
    "latency": "Time per token (sec), lower is better",
}


def plot(
    features: List[str],
    output_dir: str = "outputs",
    title: Optional[str] = None,
    perplexity_limit: Optional[float] = None,
    skip_first: int = 100,
):
    output_dir = Path(output_dir)

    fig, ax = plt.subplots()
    ax.set_xlabel("Input Sequence Length")

    for feature_i, feature in enumerate(features):
        # If we already plotted on this ax, make a new one
        if feature_i:
            ax = ax.twinx()

        for file in output_dir.glob("*.csv"):
            experiment = file.stem
            df = pd.read_csv(file)
            X = df["input_length"][skip_first:]
            Y = df[FEATURE_DF_MAP[feature]][skip_first:]
            if feature == "perplexity":
                Y = np.log(Y)
            if feature == "latency":
                poly = np.polyfit(X, Y, 20)
                poly_y = np.poly1d(poly)(X)
                ax.plot(X, poly_y, FEATURE_STYLE_MAP[feature], label=f"{experiment} {feature}")
            else:
                ax.plot(X, Y, FEATURE_STYLE_MAP[feature], label=f"{experiment} {feature}")

        ax.set_ylabel(FEATURE_LABEL_MAP[feature])
        if perplexity_limit and feature == "perplexity":
            ax.set_ylim(top=min(ax.get_ylim()[1], perplexity_limit))

        ax.legend(loc=[1, 2, 7][feature_i])  # upper right, upper left, center right

    ax.set_title(title.replace("\\n", "\n") or "Log perplexity as a function of input lengths")
    fig.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser()
    # Where csv files have been logged
    parser.add_argument("--output_dir", type=str, default="benchmark/outputs")
    parser.add_argument(
        "--features", choices=["perplexity", "memory", "latency"], nargs="+", default=["perplexity", "memory"]
    )
    parser.add_argument("--title", type=str, default="Log perplexity as a function of input lengths")
    parser.add_argument("--log_perplexity_limit", type=float, default=5.0)
    # Perplexity starts a bit unstable, so we skip the start
    parser.add_argument("--skip_first", type=int, default=100)
    parser.add_argument("--figure_dir", type=str, default="perplexity.svg")

    args = parser.parse_args()

    figure = plot(
        args.features,
        output_dir=args.output_dir,
        title=args.title,
        perplexity_limit=args.log_perplexity_limit,
        skip_first=args.skip_first,
    )

    figure.savefig(args.figure_dir)
    # plt.show()


if __name__ == "__main__":
    main()
