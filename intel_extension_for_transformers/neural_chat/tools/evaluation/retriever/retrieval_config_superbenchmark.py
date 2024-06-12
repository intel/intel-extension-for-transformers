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

import argparse
import os
import subprocess
import jsonlines
import yaml

def main():
    if os.path.exists("result_retrieval.jsonl"):
        os.remove("result_retrieval.jsonl")
    script_path = 'retrieval_benchmark.sh'

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yaml")
    args = parser.parse_args()

    data = read_yaml_file(args.config_path)
    data = {k: [str(item) for item in v] if isinstance(v, list) else str(v) for k, v in data.items()}
    args = parser.parse_args()

    data = read_yaml_file(args.config_path)
    data = {k: [str(item) for item in v] if isinstance(v, list) else str(v) for k, v in data.items()}
    arg1 = data['index_file_jsonl_path']
    arg2 = data['query_file_jsonl_path']
    arg3 = data['vector_database']
    arg4 = data['embedding_model']
    arg5 = data['llm_model']
    arg6 = data['reranker_model']
    arg7_list = data['retrieval_type']
    arg8_list = data['polish']
    arg9_list = data['search_type']
    arg10_list = data['k']
    arg11_list = data['fetch_k']
    arg12_list = data['score_threshold']
    arg13_list = data['top_n']
    arg14_list = data['enable_rerank']

    for arg7 in arg7_list:
        print('--'*1 +'retrieval_type',arg7)
        for arg8 in arg8_list:
            print('--'*2 +'polish',arg8)
            for arg9 in arg9_list:
                print('--'*3 +'search_type',arg9)
                for arg10 in arg10_list:
                    print('--'*4 +'k',arg10)
                    for arg11 in arg11_list:
                        print('--'*5 +'fetch_k',arg11)
                        for arg12 in arg12_list:
                            print('--'*6 +'score_threshold',arg12)
                            for arg13 in arg13_list:
                                print('--'*7 +'top_n',arg13)
                                for arg14 in arg14_list:
                                    print('--'*8 +'enable_rerank',arg14)
                                    # try:
                                    subprocess.run(['bash',
                                                    script_path,
                                                    '--index_file_jsonl_path='+arg1,
                                                    '--query_file_jsonl_path='+arg2,
                                                    '--vector_database='+arg3,
                                                    '--embedding_model='+arg4,
                                                    '--llm_model='+arg5,
                                                    '--reranker_model='+arg6,
                                                    '--retrieval_type='+arg7,
                                                    '--polish='+arg8,
                                                    '--search_type='+arg9,
                                                    '--k='+arg10,
                                                    '--fetch_k='+arg11,
                                                    '--score_threshold='+arg12,
                                                    '--top_n='+arg13,
                                                    '--enable_rerank='+arg14],
                                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    file_jsonl_path='result_retrieval.jsonl'

    MRR_list = []
    Hit_list = []

    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            MRR=stu["MRR"]
            Hit=stu["Hit"]
            MRR_list.append(MRR)
            Hit_list.append(Hit)

    MRR_line_number_list = [i for i, v in enumerate(MRR_list) if v == max(MRR_list)]
    Hit_line_number_list = [i for i, v in enumerate(Hit_list) if v == max(Hit_list)]

    line=0
    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            if line in MRR_line_number_list:
                print('max_MRR',stu)
            line+=1

    line=0
    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            if line in Hit_line_number_list:
                print('max_Hit',stu)
            line+=1

def read_yaml_file(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

if __name__ == '__main__':
    main()
