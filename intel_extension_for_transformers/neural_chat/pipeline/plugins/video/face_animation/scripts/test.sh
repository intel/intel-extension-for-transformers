
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

# ### some test command before commit.
# python inference.py --preprocess crop --size 256
# python inference.py --preprocess crop --size 512

# python inference.py --preprocess extcrop --size 256
# python inference.py --preprocess extcrop --size 512

# python inference.py --preprocess resize --size 256
# python inference.py --preprocess resize --size 512

# python inference.py --preprocess full --size 256
# python inference.py --preprocess full --size 512

# python inference.py --preprocess extfull --size 256
# python inference.py --preprocess extfull --size 512

python inference.py --preprocess full --size 256 --enhancer gfpgan
python inference.py --preprocess full --size 512 --enhancer gfpgan

python inference.py --preprocess full --size 256 --enhancer gfpgan --still
python inference.py --preprocess full --size 512 --enhancer gfpgan --still
