SET conda_env_name=windows_build
SET python_version=3.10
cd ../../..
echo "list curent directory"
dir
FOR /F %%i IN ('conda info -e ^| find /c "%conda_env_name%"') do SET CONDA_COUNT=%%i
if %CONDA_COUNT% NEQ 0 (
    CALL conda env remove --name "%conda_env_name:"=%"
    IF %ERRORLEVEL% NEQ 0 (
        echo "Could not remove conda environment."
        exit 1
    )
)

CALL conda create python=%python_version% -y -n %conda_env_name%
IF %ERRORLEVEL% NEQ 0 (
    echo "Could not create new conda environment."
    exit 1
)
CALL conda activate %conda_env_name%
CALL pip install -U pip
echo "Installing requirements for validation scripts..."
CALL pip install -r requirements.txt
echo "pip list all the components------------->"
CALL pip list
echo "------------------------------------------"
IF %ERRORLEVEL% NEQ 0 (
    echo "Could not install requirements."
    exit 1
)

git submodule update --init --recursive
python setup.py sdist bdist_wheel
IF %ERRORLEVEL% NEQ 0 (
    echo "Could not build binary."
    exit 1
)