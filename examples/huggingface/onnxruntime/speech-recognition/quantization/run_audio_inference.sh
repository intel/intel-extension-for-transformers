#!/bin/bash
set -x

function main {

  init_params "$@"
  run_audio_inference

}

# init params
function init_params {
  audio_path=../../../../../intel_extension_for_transformers/neural_chat/assets/audio/sample.wav
  script="run_whisper.py"
  for var in "$@"
  do
    case $var in
      --config=*)
          config=$(echo $var |cut -f2 -d=)
      ;;
      --audio_path=*)
          audio_path=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}


# run_audio_inference
function run_audio_inference {

    python -u ${script} \
        --model_name_or_path ${config} \
        --input_model ${input_model} \
        --audio_path ${audio_path} \
        --benchmark \
        --audio_test
}

main "$@"
