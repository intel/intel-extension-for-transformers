# Build Intel Extension for Transformers with Docker Image
## Prepare Docker Image
For linux usage, we recomment using image: quay.io/pypa/manylinux2014_x86_64
```shell
docker pull quay.io/pypa/manylinux2014_x86_64
docker run -i -t --name="xTransformers" --hostname="xTransformers" -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY quay.io/pypa/manylinux2014_x86_64
```

## Set Up Environment
Install GCC and G++ for binary build.
```shell
yum update && yum install -y wget git gcc
```

To use different python version, recommend to install minconda for package control
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Mini*
```

## Build Binary
Use conda create dependent python environment, download [intel-extension-for-transformer](https://github.com/intel/intel-extension-for-transformers), then build .whl.
```shell
conda create -n python38 python=3.8 -y
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
pip install --upgrade build
python3 -m build --sdist --wheel
cp dist/intel_extension_for_transformers*.whl ./
```

If you only use backends, just set environment variable `BACKENDS_ONLY=1`. The new package is named intel_extension_for_transformers_backends.
```shell
BACKENDS_ONLY=1 python3 -m build --sdist --wheel
```
>**Note**: Please check either intel_extension_for_transformers or intel_extension_for_transformers_backends installed in env to prevent possible confilcts. You can pip uninstall intel_extension_for_transformers/intel_extension_for_transformers_backends before installing.