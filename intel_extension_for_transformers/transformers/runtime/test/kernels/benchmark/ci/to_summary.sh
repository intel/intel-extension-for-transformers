#!/bin/bash
#===============================================================================
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

cat $1 |
    gawk 'BEGIN {
        total_labels = 0;
        total_cases = 0;
    }
    {
        if ($1 == ">>>>>>>###") {
            for (i=2; i<=NF; i++) {
                curr_label[i-1] = $i;
                if (!($i in labels_dict)) labels_dict[$i]=(++total_labels);
            } 
        } else if ($1 == ">>>>>>>>>>") {
            total_cases ++;
            for (i=2; i<=NF; i++) {
                result[total_cases,"label",curr_label[i-1]] = $i;
            }
            getline out_acc;
            result[total_cases,"acc"] = out_acc == "result correct" ? "correct" : "incorrect";

            getline out_perf;
            result[total_cases,"perf"] = out_perf;
        } 
    } END {
        PROCINFO["sorted_in"]="@val_num_asc";
        for (label in labels_dict) {
            printf("%s;", label);
        }
        printf("acc;perf;\n")
        for (i_case=1; i_case <= total_cases; i_case++) {
            for (label in labels_dict) {
                if ((i_case,"label",label) in result) 
                    printf("%s;", result[i_case,"label",label]);
                else
                    printf("na;");
            }
            if ((i_case, "acc") in result)
                printf("%s;", result[i_case, "acc"]);
            else
                printf("%s;\n", "na");
            if ((i_case, "perf") in result)
                printf("%s;\n", result[i_case, "perf"]);
            else
                printf("%s;\n", "na");
        }
    }'
