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
log_dir=${workspace}/.github/workflows/script/formatScan


echo "---Updating definition (DAT) files ---"
DEFS_URL=https://update.nai.com/products/commonupdater/current/vscandat1000/dat/0000
echo "Finding latest defs at $DEFS_URL/avvdat.ini..." \
 && wget -q $DEFS_URL/avvdat.ini \
 && echo "SUCCESS" || fail

inifile="avvdat.ini"
filename=`awk -F"=" '$2 ~ /avvdat.*zip/ { print $2 } ' $inifile`
filename2="$(echo -e "${filename}" | tr -d '[:space:]')"

if [ -z "$filename2" ]
then
      echo "Cannot get defs information from INI file:"
      cat $inifile
      fail
fi

echo "Downloading latest defs from $DEFS_URL/$filename2..." \
 && wget -q $DEFS_URL/$filename2 \
 && echo "SUCCESS" || fail

echo "Extracting latest defs..." \
 && unzip -o $filename2 -d ${workspace}/uvscan \
 && echo "SUCCESS" || fail

echo "--- Scanning ---"
ENV_SCAN_OPTS="--analyze --mime --program --recursive --unzip --threads 4 --summary --verbose --html=${workspace}/.github/workflows/script/formatScan/report.html"
echo "Scan Options: $ENV_SCAN_OPTS"

rm -r ${workspace}/avvdat*
rm -r ${workspace}/.git
uvscan $ENV_SCAN_OPTS ${workspace} 2>&1 | tee ${log_dir}/trellix.log


if [[ $(grep "Possibly Infected" ${log_dir}/trellix.log | sed 's/[^0-9]//g') != 0 ]]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and check error details." && $RESET
    exit 1
fi

$BOLD_PURPLE && echo "Congratulations, Trellix Scan passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0