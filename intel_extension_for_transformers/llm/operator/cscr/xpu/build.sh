rm -rf build
mkdir build
cd build
cmake ..
make -j
cd ..
python setup_2.py install
