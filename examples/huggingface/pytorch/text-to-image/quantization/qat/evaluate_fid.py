#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import copy
import logging
import os
import time
import numpy as np
import pathlib

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.datasets as dset
import torchvision.transforms as transforms
from text2images import StableDiffusionPipelineMixedPrecision

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="", help="Model path")
    parser.add_argument("--int8_model_path", type=str, default="", help="INT8 model path")
    parser.add_argument("--dataset_path", type=str, default="", help="COCO2017 dataset path")
    parser.add_argument("--output_dir", type=str, default=None,help="output path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument('--precision', type=str, default="fp32", help='precision: fp32, bf16, int8, int8-bf16')
    parser.add_argument('-i', '--iterations', default=-1, type=int, help='number of total iterations to run')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='ccl', type=str, help='distributed backend')

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    logging.info(f"Parameters {args}")

    # CCL related
    os.environ['MASTER_ADDR'] = str(os.environ.get('MASTER_ADDR', '127.0.0.1'))
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        print("World size: ", args.world_size)

    args.distributed = args.world_size > 1
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

    # load model
    pipe = StableDiffusionPipelineMixedPrecision.from_pretrained(args.model_name_or_path)
    pipe.HIGH_PRECISION_STEPS = 5

    # data type
    if args.precision == "fp32":
        print("Running fp32 ...")
        dtype=torch.float32
    elif args.precision == "bf16":
        print("Running bf16 ...")
        dtype=torch.bfloat16
    elif args.precision == "int8" or args.precision == "int8-bf16":
        print(f"Running {args.precision} ...")
        if args.precision == "int8-bf16":
            unet_bf16 = copy.deepcopy(pipe.unet).to(device=pipe.unet.device, dtype=torch.bfloat16)
            pipe.unet_bf16 = unet_bf16
        from quantization_modules import load_int8_model
        pipe.unet = load_int8_model(pipe.unet, args.int8_model_path, "fake" in args.int8_model_path)
    else:
        raise ValueError("--precision needs to be the following:: fp32, bf16, fp16, int8, int8-bf16")

    # pipe.to(dtype)
    if args.distributed:
        torch.distributed.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        print("Rank and world size: ", torch.distributed.get_rank()," ", torch.distributed.get_world_size())
        # print("Create DistributedDataParallel in CPU")
        # pipe = torch.nn.parallel.DistributedDataParallel(pipe)

    # prepare dataloader
    val_coco = dset.CocoCaptions(root = '{}/val2017'.format(args.dataset_path),
                                 annFile = '{}/annotations/captions_val2017.json'.format(args.dataset_path),
                                 transform=transforms.Compose([transforms.Resize((512, 512)), transforms.PILToTensor(), ]))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_coco, shuffle=False)
    else:
        val_sampler = None

    val_dataloader = torch.utils.data.DataLoader(val_coco,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 sampler=val_sampler)

    print("Running accuracy ...")
    # run model
    if args.distributed:
        torch.distributed.barrier()
    fid = FrechetInceptionDistance(normalize=True)
    for i, (images, prompts) in enumerate(val_dataloader):
        prompt = prompts[0][0]
        real_image = images[0]
        print("prompt: ", prompt)
        if args.precision == "bf16":
            context = torch.cpu.amp.autocast(dtype=dtype)
            with context, torch.no_grad():
                output = pipe(prompt, generator=torch.manual_seed(args.seed), output_type="numpy").images
        else:
            with torch.no_grad():
                output = pipe(prompt, generator=torch.manual_seed(args.seed), output_type="numpy").images

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            image_name = time.strftime("%Y%m%d_%H%M%S")
            Image.fromarray((output[0] * 255).round().astype("uint8")).save(f"{args.output_dir}/fake_image_{image_name}.png")
            Image.fromarray(real_image.permute(1, 2, 0).numpy()).save(f"{args.output_dir}/real_image_{image_name}.png")

        fake_image = torch.tensor(output[0]).unsqueeze(0).permute(0, 3, 1, 2)
        real_image = real_image.unsqueeze(0) / 255.0

        fid.update(real_image, real=True)
        fid.update(fake_image, real=False)

        if args.iterations > 0 and i == args.iterations - 1:
            break

    if args.distributed:
        torch.distributed.barrier()
    print(f"FID: {float(fid.compute())}")

if __name__ == '__main__':
    main()