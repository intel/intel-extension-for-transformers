# To avoid the error: 'ChatGLMTokenizer' object has no attribute 'sp_tokenizer'
script_dir=$(dirname "${BASH_SOURCE[0]}")
pip install -r "$script_dir/common.txt" transformers==4.33.1
