#!/bin/bash
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

source /intel-extension-for-transformers/.github/workflows/script/change_color.sh
log_dir=/GenAIExamples/.github/workflows/script/formatScan

find . -type f \( -name "Dockerfile*" \) -print -exec hadolint --ignore DL3006 --ignore DL3007 --ignore DL3008 {} \; 2>&1 | tee ${log_dir}/hadolint.log

if [[ $(grep -c "error" ${log_dir}/hadolint.log) != 0 ]]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and check error details." && $RESET
    exit 1
fi

$BOLD_PURPLE && echo "Congratulations, Hadolint check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0