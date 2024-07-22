# Copyright (c) 2024 Intel Corporation
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
import torch

class PruneConfig(dict):
    def __init__(self, real_drop=True):
        self.real_drop = real_drop

class KVPruner:
    def __init__(self, prune_config) -> None:
        self._past_length = 0
        self.prune_kv_cache_size = None

    def self_attn_init(self, module):
        pass

    def prune(self, module, query_states, key_states, value_states, **kwargs):
        pass

    def before_generate(self, model, inputs, *args, **kwargs):
        self.past_length = 0

    def after_generate(self, model, inputs, *args, **kwargs):
        pass

    def get_mask(self, model, **kwargs):
        pass

    @property
    def past_length(self):
        return self._past_length

    @past_length.setter
    def past_length(self, value):
        self._past_length = value

    def remove_repeat_kv(self, kv_tensor, n_rep):
        if n_rep == 1:
            return kv_tensor
        drop_mask = torch.tensor(
            [True if i % n_rep == 0 else False for i in range(0, kv_tensor.size(1))]
        ).repeat(kv_tensor.size(0), 1).to(kv_tensor.device)
        new_shape = list(kv_tensor.shape)
        new_shape[1] = int(new_shape[1] / n_rep)
        kv_tensor = kv_tensor[drop_mask].view(new_shape)
        return kv_tensor