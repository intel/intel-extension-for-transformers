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
    if os.path.exists("result_ragas.jsonl"):
        os.remove("result_ragas.jsonl")
    script_path = 'ragas_benchmark.sh'

    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--use_openai_key", default=False, action='store_true')
    parser.add_argument("--vector_database", type=str, default="Chroma")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--llm_model", type=str)
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-large")

    args = parser.parse_args()

    arg1 = args.ground_truth_file
    arg2 = args.input_path
    arg3 = str(args.use_openai_key)
    arg4 = args.vector_database
    arg5 = args.embedding_model
    arg6 = args.llm_model
    arg7 = args.reranker_model

    arg8_list = ['default','child_parent','bm25']
    arg9_list = ['True','False']
    arg10_list = ['similarity','mmr','similarity_score_threshold']
    arg11_list = ['1', '3', '5']
    arg12_list = ['5', '10', '20']
    arg13_list = ['0.3','0.5','0.7']
    arg14_list = ['1','3', '5','10']
    arg15_list = ['True','False']
    arg16_list = ['256','512', '768','1024']
    arg17_list = ['0.01','0.05', '0.1','0.3','0.5','0.7']
    arg18_list = ['1','3', '10','20']
    arg19_list = ['0.1','0.3', '0.5','0.7']
    arg20_list = ['1.0','1.1', '1.3','1.5','1.7']
    arg21_list = ['1','3', '10','20']
    arg22_list = ['True','False']

    for arg8 in arg8_list:
        print('--'*1 +'retrieval_type',arg8)
        for arg9 in arg9_list:
            print('--'*2 +'polish',arg9)
            for arg10 in arg10_list:
                print('--'*3 +'search_type',arg10)
                for arg11 in arg11_list:
                    print('--'*4 +'k',arg11)
                    for arg12 in arg12_list:
                        print('--'*5 +'fetch_k',arg12)
                        for arg13 in arg13_list:
                            print('--'*6 +'score_threshold',arg13)
                            for arg14 in arg14_list:
                                print('--'*7 +'top_n',arg14)
                                for arg15 in arg15_list:
                                    print('--'*8 +'enable_rerank',arg15)
                                    for arg16 in arg16_list:
                                        print('--'*9 +'max_chuck_size',arg16)
                                        for arg17 in arg17_list:
                                            print('--'*10 +'temperature',arg17)
                                            for arg18 in arg18_list:
                                                print('--'*11 +'top_k',arg18)
                                                for arg19 in arg19_list:
                                                    print('--'*12 +'top_p',arg19)
                                                    for arg20 in arg20_list:
                                                        print('--'*13 +'repetition_penalty',arg20)
                                                        for arg21 in arg21_list:
                                                            print('--'*14 +'num_beams',arg21)
                                                            for arg22 in arg22_list:
                                                                print('--'*15 +'do_sample',arg22)
                                                                subprocess.run(['bash',
                                                                                script_path,
                                                                                '--ground_truth_file='+arg1,
                                                                                '--input_path='+arg2,
                                                                                '--use_openai_key='+arg3,
                                                                                '--vector_database='+arg4,
                                                                                '--embedding_model='+arg5,
                                                                                '--llm_model='+arg6,
                                                                                '--reranker_model='+arg7,
                                                                                '--retrieval_type='+arg8,
                                                                                '--polish='+arg9,
                                                                                '--search_type='+arg10,
                                                                                '--k='+arg11,
                                                                                '--fetch_k='+arg12,
                                                                                '--score_threshold='+arg13,
                                                                                '--top_n='+arg14,
                                                                                '--enable_rerank='+arg15,
                                                                                '--max_chuck_size='+arg16,
                                                                                '--temperature='+arg17,
                                                                                '--top_k='+arg18,
                                                                                '--top_p='+arg19,
                                                                                '--repetition_penalty='+arg20,
                                                                                '--num_beams='+arg21,
                                                                                '--do_sample='+arg22],
                                                                                stdout=subprocess.DEVNULL,
                                                                                stderr=subprocess.DEVNULL)

    file_jsonl_path='result_ragas.jsonl'

    answer_relevancy_average_list = []
    faithfulness_average_list = []
    context_recall_average_list = []
    context_precision_average_list = []

    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            answer_relevancy_average=stu["answer_relevancy_average"]
            faithfulness_average=stu["faithfulness_average"]
            context_recall_average=stu["context_recall_average"]
            context_precision_average=stu["context_precision_average"]

            answer_relevancy_average_list.append(answer_relevancy_average)
            faithfulness_average_list.append(faithfulness_average)
            context_recall_average_list.append(context_recall_average)
            context_precision_average_list.append(context_precision_average)

    answer_relevancy_average_line_number_list = [i for i, v in enumerate(answer_relevancy_average_list) \
                                                 if v == max(answer_relevancy_average_list)]
    faithfulness_average_line_number_list = [i for i, v in enumerate(faithfulness_average_list) \
                                             if v == max(faithfulness_average_list)]
    context_recall_average_line_number_list = [i for i, v in enumerate(context_recall_average_list) \
                                               if v == max(context_recall_average_list)]
    context_precision_average_line_number_list = [i for i, v in enumerate(context_precision_average_list) \
                                                  if v == max(context_precision_average_list)]

    line=0
    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            if line in answer_relevancy_average_line_number_list:
                print('max_answer_relevancy_average',stu)
            line+=1

    line=0
    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            if line in faithfulness_average_line_number_list:
                print('max_faithfulness_average',stu)
            line+=1

    line=0
    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            if line in context_recall_average_line_number_list:
                print('max_context_recall_average',stu)
            line+=1

    line=0
    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            if line in context_precision_average_line_number_list:
                print('max_context_precision_average',stu)
            line+=1


if __name__ == '__main__':
    main()
