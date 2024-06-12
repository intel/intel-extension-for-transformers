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

set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  retrieval_type='default'
  search_type="similarity"
  k=1
  fetch_k=5
  score_threshold=0.3
  top_n=1
  max_chuck_size=256
  temperature=0.01
  top_k=1
  top_p=0.1
  repetition_penalty=1.0
  num_beams=1

  for var in "$@"
  do
    case $var in
     --ground_truth_file=*)
          ground_truth_file=$(echo $var |cut -f2 -d=)
      ;;
      --input_path=*)
          input_path=$(echo $var |cut -f2 -d=)
      ;;
      --use_openai_key=*)
          use_openai_key=$(echo $var |cut -f2 -d=)
      ;;
      --vector_database=*)
          vector_database=$(echo $var |cut -f2 -d=)
      ;;
      --embedding_model=*)
          embedding_model=$(echo $var |cut -f2 -d=)
      ;;
      --llm_model=*)
          llm_model=$(echo $var |cut -f2 -d=)
      ;;
      --reranker_model=*)
          reranker_model=$(echo ${var} |cut -f2 -d=)
      ;;
      --retrieval_type=*)
          retrieval_type=$(echo $var |cut -f2 -d=)
      ;;
      --polish=*)
          polish=$(echo $var |cut -f2 -d=)
      ;;
      --search_type=*)
          search_type=$(echo $var |cut -f2 -d=)
      ;;
      --k=*)
          k=$(echo $var |cut -f2 -d=)
      ;;
      --fetch_k=*)
          fetch_k=$(echo $var |cut -f2 -d=)
      ;;
      --score_threshold=*)
          score_threshold=$(echo ${var} |cut -f2 -d=)
      ;;
      --top_n=*)
          top_n=$(echo ${var} |cut -f2 -d=)
      ;;
      --enable_rerank=*)
          enable_rerank=$(echo $var |cut -f2 -d=)
      ;;
      --max_chuck_size=*)
          max_chuck_size=$(echo $var |cut -f2 -d=)
      ;;
      --temperature=*)
          temperature=$(echo $var |cut -f2 -d=)
      ;;
      --top_k=*)
          top_k=$(echo $var |cut -f2 -d=)
      ;;
      --top_p=*)
          top_p=$(echo $var |cut -f2 -d=)
      ;;
      --repetition_penalty=*)
          repetition_penalty=$(echo ${var} |cut -f2 -d=)
      ;;
      --num_beams=*)
          num_beams=$(echo ${var} |cut -f2 -d=)
      ;;
      --do_sample=*)
          do_sample=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}



# run_benchmark
function run_benchmark {

    if [[ ${use_openai_key} == True ]]; then
        use_openai_key="--use_openai_key"
    else
        use_openai_key=""
    fi
    if [[ ${polish} == True ]]; then
        polish="--polish"
    else
        polish=""
    fi
    if [[ ${enable_rerank} == True ]]; then
        enable_rerank="--enable_rerank"
    else
         enable_rerank=""
    fi
    if [[ ${do_sample} == True ]]; then
        do_sample="--do_sample"
    else
        do_sample=""
    fi

    python -u ./ragas_evaluation_benchmark.py \
        --ground_truth_file ${ground_truth_file} \
        --input_path ${input_path} \
        ${use_openai_key} \
        --vector_database ${vector_database} \
        --embedding_model ${embedding_model} \
        --llm_model ${llm_model} \
        --reranker_model ${reranker_model} \
        --retrieval_type ${retrieval_type} \
        ${polish} \
        --search_type ${search_type} \
        --k ${k} \
        --fetch_k ${fetch_k} \
        --score_threshold ${score_threshold} \
        --top_n ${top_n} \
        ${enable_rerank} \
        --max_chuck_size ${max_chuck_size} \
        --temperature ${temperature} \
        --top_k ${top_k} \
        --top_p ${top_p} \
        --repetition_penalty ${repetition_penalty} \
        --num_beams ${num_beams} \
        ${do_sample} 
}

main "$@"
