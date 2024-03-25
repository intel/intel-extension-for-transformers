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

def main():
    if os.path.exists("result.jsonl"):
        os.remove("result.jsonl")
    script_path = 'evaluate_retrieval_benchmark.py'

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file_jsonl_path", type=str)
    parser.add_argument("--query_file_jsonl_path", type=str)
    parser.add_argument("--vector_database", type=str, default="Chroma")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--llm_model", type=str)
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-large")

    args = parser.parse_args()

    arg1 = args.index_file_jsonl_path
    arg2 = args.query_file_jsonl_path
    arg3 = args.vector_database
    arg4 = args.embedding_model
    arg5 = args.retrieval_type
    arg6 = args.llm_model

    arg7_list = ['default','child_parent','bm25']
    arg8_list = ['True','False']
    arg9_list = ['similarity','mmr','similarity_score_threshold']
    arg10_list = ['1', '3', '5']
    arg11_list = ['5', '10', '20']
    arg12_list = ['0.3','0.5','0.7']
    arg13_list = ['1','3', '5','10']
    arg14_list = ['True','False']

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
                                    subprocess.run(['python',
                                                    script_path,
                                                    '--index_file_jsonl_path', arg1,
                                                    '--query_file_jsonl_path', arg2,
                                                    '--vector_database', arg3,
                                                    '--embedding_model', arg4,
                                                    '--llm_model', arg5,
                                                    '--reranker_model', arg6,
                                                    '--retrieval_type', arg7,
                                                    '--polish', arg8,
                                                    '--search_type', arg9,
                                                    '--k', arg10,
                                                    '--fetch_k', arg11,
                                                    '--score_threshold', arg12,
                                                    '--top_n', arg13,
                                                    '--enable_rerank', arg14],
                                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    file_jsonl_path='result.jsonl'

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

if __name__ == '__main__':
    main()
