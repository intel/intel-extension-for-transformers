# Build Intel Extension for Transformers with Docker Image
## Prepare Docker Image
For linux usage, we recomment using image: quay.io/pypa/manylinux2014_x86_64
```
docker pull quay.io/pypa/manylinux2014_x86_64
docker run -i -t --name="xTransformers" --hostname="xTransformers" -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY quay.io/pypa/manylinux2014_x86_64
```

## Set Up Environment
Install GCC and G++ for binary build.
```
yum update && yum install -y wget git gcc
```

To use different python version, recommend to install minconda for package control
```
wget  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Mini*
```

## Build Binary
Use conda create dependent python environment, download [intel-extension-for-transformer](https://github.com/intel/intel-extension-for-transformers), then build .whl. Finally, use auditwheel to make it feasible across linux OS
```
conda create -n python38 python=3.8 -y
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers
git submodule update --init --recursive
[[ -f requirements.txt ]] && pip install -r requirements.txt
python setup.py sdist bdist_wheel
pip install auditwheel==5.1.2
auditwheel repair dist/intel_extension_for_transformers*.whl
cp wheelhouse/intel_extension_for_transformers*.whl ./
```
If only use backends, just add "--backends" while installing.
```
python3 setup.py sdist bdist_wheel --backends 
```