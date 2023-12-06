//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#ifndef MODELS_H
#define MODELS_H

#include "models/model_utils/model_types.h"

struct IModel {
  virtual void init(const char* path_model, model_context* ctx, int n_gpu_layers, bool use_mmap, bool use_mlock,
                    bool vocab_only) = 0;
  virtual void load(model_context* ctx, model_progress_callback progress_callback,
                    void* progress_callback_user_data) = 0;
};

#endif  // MODELS_H
