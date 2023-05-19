# Enable clang-format when you do "git commit"
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${script_dir}/clang_format.sh

# Set up the environment for Intel oneAPI DPC++/C++ Compiler
# ONEAPI_INSTALL_PATH below assumes you installed to the default folder /opt/intel/oneapi
# If you customized the installation folder, please update ONEAPI_INSTALL_PATH to your custom folder
ONEAPI_INSTALL_PATH=/opt/intel/oneapi
source ${ONEAPI_INSTALL_PATH}/setvars.sh

# Export environment variables
export CC=icx
export ONEAPI_DEVICE_SELECTOR=level_zero1:*
