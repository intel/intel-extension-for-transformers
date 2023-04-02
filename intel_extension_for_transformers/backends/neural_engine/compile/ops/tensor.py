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

"""The neural engine tensor file."""

from collections import OrderedDict
from .. import graph_utils as util
import numpy as np


class Tensor(object):
    """The definition of the neural engine."""
    def __init__(self,
                 name='',
                 source_op=None,
                 dest_op=None,
                 shape=None,
                 data=None,
                 dtype=None,
                 location=None):
        """The __init__ function."""
        self._name = name
        # assume data in tensor should be numpy array
        # however, we don't assign the data diretly if the tensor is
        # const like weight when parse model
        # otherwise it will make a bloated new graph
        # but it still can be set when using the constructed new graph
        self._data = data
        if shape is not None and len(shape) == 0:
            self._shape = [1]
        else:
            self._shape = shape
        self._dtype = dtype
        if not dtype and isinstance(data, np.ndarray):
            self._dtype = util.get_data_dtype(data)
        # location in bin file if const
        self._location = location
        if source_op == None:
            self._source_op = []
        else:
            self._source_op = source_op
        if dest_op == None:
            self._dest_op = []
        else:
            self._dest_op = dest_op

    @property
    def name(self):
        """Get the tensor name."""
        return self._name

    @name.setter
    def name(self, name):
        """Name Assignment."""
        self._name = name

    @property
    def data(self):
        """Get the tensor data."""
        return self._data

    @data.setter
    def data(self, data):
        """Data assignment."""
        self._data = data
        if data is not None:
            self._dtype = util.get_data_dtype(data)

    @property
    def shape(self):
        """Get the tensor shape."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        """Shape assignment."""
        self._shape = shape

    @property
    def dtype(self):
        """Get the tensor dtype."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        """Dtype assignment."""
        self._dtype = dtype

    @property
    def location(self):
        """Get the tensor location."""
        return self._location

    @location.setter
    def location(self, location):
        """Location assignment."""
        self._location = location

    @property
    def source_op(self):
        """Get source_op."""
        return self._source_op

    @source_op.setter
    def source_op(self, source_op):
        """Source_op assignment."""
        self._source_op = source_op

    @property
    def dest_op(self):
        """Get dest_op."""
        return self._dest_op

    @dest_op.setter
    def dest_op(self, dest_op):
        """Dest_op assignment."""
        self._dest_op = dest_op

    @property
    def config(self):
        """Get the config dict in the graph."""
        conf_dict = OrderedDict()
        if self._dtype is not None:
            conf_dict['dtype'] = util.DTYPES_DICT.get(self._dtype, self._dtype)
        if self._shape is not None:
            conf_dict['shape'] = self._shape
        if self._location is not None:
            conf_dict['location'] = self._location

        return conf_dict
