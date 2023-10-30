source ../../../library/xetla/tools/scripts/env.sh
rm -rf build
mkdir build
cd build
cmake ..
make -j
cd ..
python setup.py install
