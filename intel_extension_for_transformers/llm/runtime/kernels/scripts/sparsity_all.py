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
import os
import struct

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def cal_sparse_pytorch(tensor, block):
    non_zero = {}
    for i in tensor.nonzero():
        block_x = int(i[0]) // block[0]
        block_y = int(i[1]) // block[1]
        if block_x not in non_zero:
            non_zero[block_x] = set()
        non_zero[block_x].add(block_y)
    block_non_zero = 0
    hotmap = np.zeros((tensor.shape[0] // block[0], tensor.shape[1] // block[1]))
    for k, v in non_zero.items():
        block_non_zero += len(v)
        for i in v:
            hotmap[k][i] = 1.0
    return block_non_zero,hotmap

block = [4, 1]
type_map = {"fp32": ("f", 4), "s8": ("b", 1), "u8": ("B", 1), "s32": ("i", 4)}
#Settings
parser=argparse.ArgumentParser(description='Visualize pytorch model or nn model')
parser.add_argument(
    '--modeltype',
    type=int,
    default=0,
    help="modeltype: 0 pytorch model, 1 nn model"
    
)
parser.add_argument(
    '--path',
    type=str,
    help="please input your path"
)
args=parser.parse_args()

if __name__ =='__main__':
    if args.modeltype==0:
        print("-"*100+"visulize pytorch model"+"-"*100)
        print("model path:"+str(args.path))
        model_file=''
        for root, dirs, files in os.walk(args.path): 
            for file_name in files: #search .bin file
                if '.bin' in file_name:
                    model_file=file_name
            if model_file=='':
                assert "Must contain .bin file in this directory!"   
            path_model=args.path+"/"+model_file       
            checkpoint=torch.load(path_model,map_location="cpu")
            offset_path=args.path+"/pytorch_model_hotmaps/"
            if not  os.path.exists(offset_path):
                os.mkdir(offset_path)
            with open(args.path+"output4x1.txt", "w") as f:
                block = [4, 1]#different to neural engine
                indx=0
                for name in checkpoint:
                    if "weight" in name and len((checkpoint[name]).shape)>1: #and name.find("weight") > 0
                        tensor = checkpoint[name]
                        tensor_size = int(tensor.numel()) #get number of element 
                        elt_zero = tensor_size - tensor.nonzero().size(0)
                        elt_sparse = elt_zero / tensor_size
                        if len(tensor.size()) > 1:
                            if tensor.size()[0] % block[0] == 0 and tensor.size()[1] % block[1] == 0:
                                block_num=tensor_size//block[0]//block[1]
                                block_one, hotmap = cal_sparse_pytorch(tensor, block)
                                block_zero = (tensor_size // block[0] // block[1]) - block_one
                                block_sparse = block_zero / (tensor_size / block[0] / block[1])
                                print(name, "\tElement",elt_zero, elt_sparse,"\tBlock", block_zero, block_sparse)
                                f.write("{}:{},{}\t{},{}\n".format(name, elt_zero, elt_sparse, block_zero,
                                                                    block_sparse))
                                block_sparse = block_zero / block_num
                                if block_sparse>0.5:
                                    plt.clf()
                                    plt.axis("off")
                                    plt.imshow(hotmap, interpolation='nearest')
                                    plt.colorbar()
                                    plt.title("tensor size:"+str(hotmap.shape)+" bs:"+str(block_sparse),loc='left')
                                    plt.savefig(offset_path+str(indx)+"-"+str(name)+".jpg")
                                    indx+=1
                            else:
                                print(name, elt_zero, elt_sparse, tensor.size())
                                f.write("{}:{},{}\t{}".format(name, elt_zero, elt_sparse, tensor.size()))

        
    if args.modeltype==1:
        print("-"*100+"visulize nn model"+"-"*100)
        print("model path:"+str(args.path))
        ir_path=args.path
        if not os.path.exists(ir_path+"conf.yaml"):
            assert "your directory must contain confg.yaml file"
        if not os.path.exists(ir_path+"model.bin"):
            assert "your directory must contain model.bin file"
        offset_path=ir_path+"/nn_model_hotmaps/"
        if not  os.path.exists(offset_path):
            os.mkdir(offset_path)
        with open(os.path.join(ir_path, "conf.yaml")) as conf_file:
            conf = yaml.load(conf_file, Loader=yaml.FullLoader)
            operators = conf["model"]["operator"]
            inputs = operators["input_data"]["output"]
            indx=0
            with open(os.path.join(ir_path, "model.bin"), "rb") as weight_file:
                for _, value in filter(lambda tmp: tmp[-1]["type"] == "InnerProduct",
                                    operators.items()):
                    tensors = value["input"]
                    for tensor_name in filter(lambda tensor_name: tensor_name in inputs,
                                            tensors.keys()):
                        tensor = inputs[tensor_name]
                        dtype = tensor["dtype"]
                        if dtype != "s8":
                            continue
                        shape = tensor["shape"]
                        if len(shape) == 2 and shape[0] % block[0] == 0 and shape[1] % block[1] == 0:
                            size = 1
                            for i in shape:
                                size *= i
                            location = tensor["location"]
                            weight_file.seek(location[0], 0)
                            data_b = weight_file.read(location[1])
                            data = []
                            for i in range(size):
                                tmp = struct.unpack_from(type_map[dtype][0], data_b,
                                                        i * type_map[dtype][1])
                                data.append(tmp[0])
                            data = np.array(data).reshape(shape)
                            elt_zero = size - data.nonzero()[0].size
                            elt_sparse = elt_zero / size
                            block_num = size // block[0] // block[1]
                            block_one, hotmap = cal_sparse_pytorch(data, block)
                            block_zero = block_num - block_one
                            block_sparse = block_zero / block_num
                            tensor_name=tensor_name.replace('/','-')
                            # print("tensor_name:"+str(tensor_name))
                            print(os.path.join(ir_path, tensor_name), elt_zero, size, elt_sparse,
                                block_zero, block_num, block_sparse)
                            if block_sparse > 0.5:
                                plt.clf()
                                plt.axis("off")
                                plt.imshow(hotmap, interpolation='nearest')
                                plt.colorbar()
                                plt.title("tensor size:"+str(hotmap.shape)+" bs:"+str(block_sparse),loc='left')
                                plt.savefig(offset_path+str(indx)+"-"+str(tensor_name)+".jpg")
                                indx=indx+1
                            break
                        else:
                            continue
