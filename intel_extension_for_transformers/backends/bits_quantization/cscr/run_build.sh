rm -rf build;
mkdir build;
cd build

cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.__path__[0])')" ..

make -j$(nproc)
