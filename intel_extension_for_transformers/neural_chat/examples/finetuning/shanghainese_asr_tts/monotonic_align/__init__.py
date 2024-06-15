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

from numpy import zeros, int32, float32
from torch import from_numpy

from .core import maximum_path_jit

def maximum_path(neg_cent, mask):
  """Numba optimized version.

  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent = neg_cent.data.cpu().numpy().astype(float32)
  path = zeros(neg_cent.shape, dtype=int32)

  t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
  t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
  maximum_path_jit(path, neg_cent, t_t_max, t_s_max)
  return from_numpy(path).to(device=device, dtype=dtype)
