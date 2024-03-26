import argparse
import os
import subprocess
import jsonlines

def main():
    if os.path.exists("result_ragas.jsonl"):
        os.remove("result_ragas.jsonl")
    script_path = 'ragas_evaluation_benchmark.py'

    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--vector_database", type=str, default="Chroma")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--llm_model", type=str)
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-large")

    args = parser.parse_args()

    arg1 = args.ground_truth_file
    arg2 = args.input_path
    arg3 = args.vector_database
    arg4 = args.embedding_model
    arg5 = args.llm_model
    arg6 = args.reranker_model

    arg7_list = ['default','child_parent','bm25']
    arg8_list = ['True','False']
    arg9_list = ['similarity','mmr','similarity_score_threshold']
    arg10_list = ['1', '3', '5']
    arg11_list = ['5', '10', '20']
    arg12_list = ['0.3','0.5','0.7']
    arg13_list = ['1','3', '5','10']
    arg14_list = ['True','False']
    arg15_list = ['256','512', '768','1024']
    arg16_list = ['0.01','0.05', '0.1','0.3','0.5','0.7']
    arg17_list = ['1','3', '10','20']
    arg18_list = ['0.1','0.3', '0.5','0.7']
    arg19_list = ['1.0','1.1', '1.3','1.5','1.7']
    arg20_list = ['1','3', '10','20']
    arg21_list = ['True','False']

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
                                    for arg15 in arg15_list:
                                        print('--'*9 +'max_chuck_size',arg15)
                                        for arg16 in arg16_list:
                                            print('--'*10 +'temperature',arg16)
                                            for arg17 in arg17_list:
                                                print('--'*11 +'top_k',arg17)
                                                for arg18 in arg18_list:
                                                    print('--'*12 +'top_p',arg18)
                                                    for arg19 in arg19_list:
                                                        print('--'*13 +'repetition_penalty',arg19)
                                                        for arg20 in arg20_list:
                                                            print('--'*14 +'num_beams',arg20)
                                                            for arg21 in arg21_list:
                                                                print('--'*15 +'do_sample',arg21)
                                                                subprocess.run(['python', 
                                                                                script_path, 
                                                                                '--ground_truth_file', arg1,
                                                                                '--input_path', arg2,
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
                                                                                '--enable_rerank', arg14,
                                                                                '--max_chuck_size', arg15,
                                                                                '--temperature', arg16,
                                                                                '--top_k', arg17,
                                                                                '--top_p', arg18,
                                                                                '--repetition_penalty', arg19,
                                                                                '--num_beams', arg20,
                                                                                '--do_sample', arg21],
                                                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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

    answer_relevancy_average_line_number_list = [i for i, v in enumerate(answer_relevancy_average_list) if v == max(answer_relevancy_average_list)]
    faithfulness_average_line_number_list = [i for i, v in enumerate(faithfulness_average_list) if v == max(faithfulness_average_list)]
    context_recall_average_line_number_list = [i for i, v in enumerate(context_recall_average_list) if v == max(context_recall_average_list)]
    context_precision_average_line_number_list = [i for i, v in enumerate(context_precision_average_list) if v == max(context_precision_average_list)]

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