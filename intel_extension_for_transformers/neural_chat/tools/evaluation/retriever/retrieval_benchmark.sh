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

  for var in "$@"
  do
    case $var in
     --index_file_jsonl_path=*)
          index_file_jsonl_path=$(echo $var |cut -f2 -d=)
      ;;
      --query_file_jsonl_path=*)
          query_file_jsonl_path=$(echo $var |cut -f2 -d=)
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
    esac
  done

}


# run_benchmark
function run_benchmark {

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

    python -u ./evaluate_retrieval_benchmark.py \
        --index_file_jsonl_path ${index_file_jsonl_path} \
        --query_file_jsonl_path ${query_file_jsonl_path} \
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

}

main "$@"
