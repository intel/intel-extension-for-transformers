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

import time
from turtle import width
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from executor_dataloader import dataloader_wrapper
import sys
import os

common_dir = os.path.join(sys.path[0], "../../../../neural_engine_utils/")
sys.path.append(common_dir)
from common import (log, Neural_Engine_base, DummyDataLoader)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    res = []
    for k in topk:
        max_k_preds = output.argsort(axis=1)[:, -k:][:, ::-1]
        correct_k = np.logical_or.reduce(max_k_preds == target, axis=1)
        res.append(correct_k.sum() / correct_k.shape[0] * 100)
    return res


def validate(val_loader, session, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, top1, top5, prefix='Test: ')

    for i, batch in enumerate(val_loader):
        start = time.time()
        inputs_onnx = {k: v.detach().numpy() for k, v in batch.items() if k != 'labels'}
        output = session.run(None, inputs_onnx)
        target = batch['labels'].numpy().reshape(-1, 1)
        # [batch_size, num_labels]
        output = output[0]
        # measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1, target.shape[0])
        top5.update(acc5, target.shape[0])
        batch_time.update(time.time() - start)

        if i % print_freq == 0:
            progress.print(i)

    print('Accuracy: {top1:.5f} Accuracy@5 {top5:.5f}'.format(top1=(top1.avg / 100),
                                                              top5=(top5.avg / 100)))


class Neural_Engine(Neural_Engine_base):

    def accuracy(self, batch_size, feature_extractor_name, data_dir):
        # load dataset
        log.info("Load dataset ......")
        eval_dataloader = dataloader_wrapper(batch_size, feature_extractor_name,
                                             data_dir).get_eval_data()

        def validate(val_loader, graph, print_freq=10):
            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(len(val_loader), batch_time, top1, top5, prefix='Test: ')

            for i, batch in enumerate(val_loader):
                start = time.time()
                inputs_onnx = {k: v.detach().numpy() for k, v in batch.items() if k != 'labels'}
                output = graph.inference([inputs_onnx['pixel_values']])
                output = list(output.values())[0]
                target = batch['labels'].numpy().reshape(-1, 1)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                top1.update(acc1, target.shape[0])
                top5.update(acc5, target.shape[0])
                batch_time.update(time.time() - start)

                if i % print_freq == 0:
                    progress.print(i)

            print('Accuracy: {top1:.5f} Accuracy@5 {top5:.5f}'.format(top1=(top1.avg / 100),
                                                                      top5=(top5.avg / 100)))

        validate(eval_dataloader, graph=self.graph, print_freq=10)

    def performance(self, batch_size, iteration, warm_up):
        if warm_up >= iteration:
            log.error("Warm up should less than iteration.")
            raise ValueError()
        # generate dummy dataset
        log.info("Generate dummy dataset ......")
        Channel = 3
        Height = 224
        Width = 224
        shape = [batch_size, Channel, Height, Width]
        dataset = DummyDataLoader(shapes=[shape, shape, shape, shape],
                                  lows=[1, 1, 1, 1],
                                  highs=[1, 3, 255, 255],
                                  dtypes=['float32', 'float32', 'float32', 'float32'],
                                  iteration=iteration)

        def compute_performance(dataset, graph, log, log_file, warm_up, batch_size):
            log.info("Start executor ......")
            duration = []
            for idx in tqdm(range(len(dataset))):
                start_time = time.time()
                for i in range(len(dataset[idx])):
                    graph.inference([dataset[idx][i]])
                end_time = time.time()
                duration.append(end_time - start_time)
            log.info("End executor ......")
            duration_w = duration[warm_up:]
            all_latency = log_file.replace('.log', '.npy')
            _, file_name = os.path.split(all_latency)
            _ = os.getcwd() + '/all_latency'
            try:
                if os.path.exists(_) == False:
                    os.mkdir(_)
            except:
                pass
            all_latency = os.path.join(_, file_name)
            All_latency = np.array(duration_w)
            np.save(all_latency, All_latency, allow_pickle=True, fix_imports=True)
            ave_latency = np.array(duration_w).mean() / batch_size
            p50_latency = np.percentile(duration_w, 50) / batch_size
            p90_latency = np.percentile(duration_w, 90) / batch_size
            p99_latency = np.percentile(duration_w, 99) / batch_size
            log.info("Batch size = {}".format(batch_size))
            log.info("P50 Latency: {:.3f} ms".format(p50_latency * 1000))
            log.info("P90 Latency: {:.3f} ms".format(p90_latency * 1000))
            log.info("P99 Latency: {:.3f} ms".format(p99_latency * 1000))
            log.info("Average Latency: {:.3f} ms".format(ave_latency * 1000))
            log.info("Throughput: {:.3f} samples/sec".format(1. / ave_latency))

        compute_performance(dataset, self.graph, log, self.log_file, warm_up, batch_size)
