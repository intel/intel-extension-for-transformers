#!/bin/bash
path=`pwd`
project_source_dir="${path}/intel_extension_for_transformers/backends/neural_engine/"
if (( $# >= 1 )); then
  echo "project source dir set to $1"
  project_source_dir=$1
fi

work_dir=$project_source_dir/third_party
echo "**   work dir is: ${work_dir}"
install_dir=${work_dir}/local_install
echo "**install dir is: ${install_dir}"


# build gflags
cd ${work_dir}/gflags
echo -e "\n-------- begin to gflags: build --------"
mkdir build && cd build
#cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${install_dir}/gflags/ ..
make -j && make install

# build glog
cd ${work_dir}/glog
# need gflags lib
export gflags_DIR=${work_dir}/gflags/build/
echo -e "\n-------- begin to glog: build --------"
mkdir build && cd build
#cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${install_dir}/glog/ ..
make -j && make install

# build yaml
cd ${work_dir}/yaml-cpp/
echo -e "\n-------- begin to yaml: build --------"
mkdir build && cd build
#cd build
cmake -DYAML_BUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${install_dir}/yaml-cpp/ ..
make -j && make install

# build onednn
cd ${work_dir}/oneDNNGraph/
echo -e "\n-------- begin to onednn: build --------"
mkdir build && cd build
#cd build
cmake -DDNNL_LIBRARY_TYPE=SHARED -DDNNL_BUILD_TESTS=0 -DCMAKE_INSTALL_PREFIX=${install_dir}/oneDNNGraph/ ..
make -j && make install


