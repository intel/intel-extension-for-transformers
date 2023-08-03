#!/bin/bash

source /intel-extension-for-transformers/.github/workflows/script/change_color.sh
log_dir=/intel-extension-for-transformers/.github/workflows/script/formatScan
cloc --include-lang=Python --csv --out=${log_dir}/cloc.csv /intel-extension-for-transformers
