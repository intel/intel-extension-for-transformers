#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0

"""Evolustion: Provide the evolustionary search for pytorch."""

import csv
import logging
import os
import random
import timeit
import numpy as np
import torch
from intel_extension_for_transformers.optimization.utils.utility import LazyImport
torchprofile = LazyImport("torchprofile")
logger = logging.getLogger(__name__)


def approx_ratio(x, n=12, l=384):
    """Get the approximation ratio."""
    s = 0
    i = l
    for _ in range(n):
        i = int(np.ceil(i * (1 - x)))  # i * x
        s += i
    return s / (n * l)

def inverse(x):
    """Get the inverse number."""
    l, r = 0, 1
    while r - l > 1e-12:
        c = (l + r) / 2
        v = approx_ratio(c)
        l, r = (c, r) if x <= v else (l, c)
    return l


def store2str(gene, macs, score, method, parents=None):
    """Store the parmaters into string."""
    store_str = f"({', '.join(f'{x:3d}' for x in gene)}):"
    store_str += f" {macs} MACs/latency"
    store_str += f" | score {score}"
    store_str += f" | method {method}"
    if parents is not None:
        store_str += f"| parent(s) {parents}"
    return store_str

class Evolution(object):
    """Class of Evolution supports for evolutionary searching."""
    def __init__(
        self,
        model,
        max_seq_length,
        device,
        evaluate,
        lower_constraint=0,
        upper_constraint=None,
        eval_metric='eval_f1'
    ):
        """Init an Evolution instance."""
        self.model = model
        self.max_seq_length = max_seq_length
        self.device = device
        self.evaluate = evaluate

        size = (1, self.max_seq_length)
        self.dummy_inputs = (
            torch.ones(size, dtype=torch.long).to(self.device),
            torch.ones(size, dtype=torch.long).to(self.device),
            torch.zeros(size, dtype=torch.long).to(self.device),
        )
        if self.model.config.model_type == "distilbert":
            self.dummy_inputs = self.dummy_inputs[:2]

        self.lower_constraint = lower_constraint
        self.upper_constraint = upper_constraint

        self.store = {}  # gene: (macs, score, method, parent(s))
        self.population = []
        self.eval_metric=eval_metric

    def load_store(self, store_file):
        """Load from a store file."""
        if not os.path.isfile(store_file):
            return
        with open(store_file, 'r') as f:
            for row in csv.reader(f, delimiter='\t'):
                row = tuple(eval(x) for x in row[:3])
                self.store[row[0]] = row[1:3] + (0, None)

    def save_store(self, store_file):
        """Save into a store file."""
        store_keys = sorted(self.store.keys(), key=lambda x: self.store[x][0])
        with open(store_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for gene in store_keys:
                writer.writerow([str(gene)] + [str(x) for x in self.store[gene]])

    def get_store(self):
        """Get store."""
        return self.store

    def set_lower_constraint(self, constraint):
        """Setter of lower constraint."""
        self.lower_constraint = constraint

    def set_upper_constraint(self, constraint):
        """Setter of upper constraint."""
        self.upper_constraint = constraint

    def save_population(self, population_file, population):
        """Save population into a file."""
        with open(population_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for gene in population:
                writer.writerow([str(gene)] + [str(x) for x in self.store[gene]])

    def _ccw(self, gene0, gene1, gene2):
        """Counterclockwise."""
        x0, y0 = self.store[gene0][:2]
        x1, y1 = self.store[gene1][:2]
        x2, y2 = self.store[gene2][:2]
        return (x0 * y1 + x1 * y2 + x2 * y0) - (x0 * y2 + x1 * y0 + x2 * y1)

    def convex_hull(self):
        """The function to calculate convex_hull."""
        hull = self.population[:2]
        for gene in self.population[2:]:
            if self.store[hull[-1]][1] >= self.store[gene][1]:
                continue
            while len(hull) >= 2 and self._ccw(hull[-2], hull[-1], gene) >= 0:
                del hull[-1]
            hull.append(gene)
        return hull

    def pareto_frontier(self):
        """The function to calculate population and are."""
        self.population = sorted(self.population, key=lambda x: self.store[x][:2])

        frontier = [self.population[0]]
        for gene in self.population[1:-1]:
            if self.store[gene][1] > self.store[frontier[-1]][1]:
                if self.store[gene][0] == frontier[-1][0]:
                    del frontier[-1]
                frontier.append(gene)
        frontier.append(self.population[-1])
        self.population = frontier

        area = 0
        for gene0, gene1 in zip(self.population[:-1], self.population[1:]):
            x0, y0 = self.store[gene0][:2]
            x1, y1 = self.store[gene1][:2]
            area += (x1 - x0) * y0
        area /= (self.upper_constraint - self.lower_constraint)
        return self.population, area

    def add_gene(self, gene, macs=None, score=None, method=0, parents=None):
        """Add gene to evolution."""
        if gene not in self.store:
            self.model.eval()
            if self.model.config.model_type == "distilbert":
                bert = self.model.distilbert
            elif self.model.config.model_type == "roberta":
                bert = self.model.roberta
            else:
                assert hasattr(self.model, "bert")
                bert = self.model.bert
            bert.set_length_config(gene)
            bert.set_output_attentions(True)
            macs = macs or torchprofile.profile_macs(self.model, args=self.dummy_inputs)
            if macs < self.lower_constraint:
                return False
            start_time = timeit.default_timer()
            eval_result = self.evaluate()
            evalTime = timeit.default_timer() - start_time
            score = score or eval_result[self.eval_metric]
            self.store[gene] = (macs, score, method, parents)
            logger.info(store2str(gene, macs, score, method, parents))

        macs = self.store[gene][0]
        if macs >= self.lower_constraint \
                and (self.upper_constraint is None or macs <= self.upper_constraint) \
                and gene not in self.population:
            self.population.append(gene)
            return True
        return False

    def mutate(self, mutation_prob, ray=False):
        """Do the mutate."""
        gene = random.choice(self.population)
        mutated_gene = ()
        for i in range(self.model.config.num_hidden_layers):
            if np.random.uniform() < mutation_prob:
                prev = (self.max_seq_length if i == 0 else mutated_gene[i - 1])
                next = (2 if i == self.model.config.num_hidden_layers - 1 else gene[i + 1])
                mutated_gene += (random.randrange(next, prev + 1),)
            else:
                mutated_gene += (gene[i],)
        return self.add_gene(mutated_gene, method=1, parents=(gene,)) if not ray else mutated_gene, (gene,)

    def crossover(self, ray=False):
        """Do the crossover."""
        gene0, gene1 = random.sample(self.population, 2)
        crossovered_gene = tuple((g0 + g1 + 1) // 2 for g0, g1 in zip(gene0, gene1))
        return self.add_gene(crossovered_gene, method=2, parents=(gene0, gene1)) if not ray else \
                 crossovered_gene, (gene0, gene1)
