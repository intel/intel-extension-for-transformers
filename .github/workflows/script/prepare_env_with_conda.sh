cd ${WORKING_DIR}
conda_env_name=$1
python_version=$2
if [[ -z "${conda_env_name}" ]] || [[ -z "${python_version}" ]]; then
    $BOLD_RED && echo "need provide with conda env name and python version" && $RESET
    exit 1
fi

conda create -n ${conda_env_name} python=${python_version} -y
source activate ${conda_env_name} || conda activate ${conda_env_name}
pip install -U pip

if [ -f "requirements.txt" ]; then
    python -m pip install --default-timeout=100 -r requirements.txt
    pip list
else
    echo "Not found requirements.txt file."
fi
