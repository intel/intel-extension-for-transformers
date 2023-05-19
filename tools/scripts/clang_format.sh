# Enable clang-format when you do "git commit"
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cp ${script_dir}/../clang-format/clang-fomat.hook ${script_dir}/../../.git/hooks/pre-commit
chmod +x ${script_dir}/../../.git/hooks/pre-commit
