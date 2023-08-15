
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

from abc import ABC, abstractmethod
from typing import List
import os
from fastchat.conversation import get_conv_template, Conversation

class BaseModel(ABC):
    def __init__(self):
        pass

    def match(self, model_path: str):
        return True

    @abstractmethod
    def load_model(self, kwargs: dict):
        pass

    @abstractmethod
    def predict_stream(self, params):
        """
        Abstract method for performing streaming prediction.
        This method must be implemented in the derived classes.
        Args:
            params: Parameters needed for prediction.
        Returns:
            predictions: Predicted results from the streaming process.
        """
        pass

    @abstractmethod
    def predict(self, params):
        """
        Abstract method for performing batch prediction.
        This method must be implemented in the derived classes.
        Args:
            params: Parameters needed for prediction.
        Returns:
            predictions: Predicted results.
        """
        pass

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")

    def register_tts(self, instance):
        self.tts = instance

    def register_asr(self, instance):
        self.asr = instance

    def register_safety_checker(self, instance):
        self.safety_checker = instance


# A global registry for all model adapters
model_adapters: List[BaseModel] = []

def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


def get_model_adapter(model_name_path: str) -> BaseModel:
    """Get a model adapter for a model_name_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_name_path)).lower()

    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModel:
            return adapter

    raise ValueError(f"No valid model adapter for {model_name_path}")
