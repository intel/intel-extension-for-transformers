#!/bin/bash
WORKING_PATH="/intel-extension-for-transformers"
for var in "$@"
do
    case $var in
        --WORKING_PATH=*)
            WORKING_PATH=$(echo $var | cut -f2 -d=);;
    esac
done

log_dir=${WORKING_PATH}/.github/workflows/script/formatScan
VAL_REPO=${WORKING_PATH}/.github/workflows/script/formatScan
REPO_DIR=${WORKING_PATH}

sed -i "s|\${VAL_REPO}|$VAL_REPO|g" ${VAL_REPO}/pyspelling_conf.yaml
sed -i "s|\${SCAN_REPO}|$REPO_DIR|g" ${VAL_REPO}/pyspelling_conf.yaml
echo "Modified config:"
cat ${VAL_REPO}/pyspelling_conf.yaml

pip install pyspelling
pyspelling -c ${VAL_REPO}/pyspelling_conf.yaml >${log_dir}/pyspelling.log

exit_code=$?
if [ ${exit_code} -ne 0 ]; then
    $BOLD_RED && echo "Pyspelling exited with non-zero exit code." && $RESET
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, check passed!" && $LIGHT_PURPLE && echo "You can click on the artifact button to see the log details." && $RESET
exit 0
