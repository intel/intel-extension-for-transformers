#i!/bin/bash

help () {
    echo ""
    echo "Help:"
    echo "$0 or $0 local"
    echo "    Build html for local test, not merge to gh-pages branch"
    echo "$0 version"
    echo "    Build for version (version.py), then merge & push to gh-pages branch"
    echo "$0 latest"
    echo "    Build for latest code, then merge & push to gh-pages branch"
}

if [ ! -n "$1" ]; then
  ACT=only_build_local
else
  if [ "$1" == "version"  ]; then
    ACT=build_version
  elif [ "$1" == "latest"  ]; then
    ACT=build_latest
  elif [ "$1" == "local"  ]; then
    ACT=only_build_local
  elif [ "$1" == "help"  ]; then
    help
    exit 0
  else
    echo "Wrong parameter \"$1\""
    help
    exit 1
  fi
fi

echo "ACT is ${ACT}"

if [ ${ACT} == "only_build_local" ]; then
  UPDATE_LATEST_FOLDER=1
  UPDATE_VERSION_FOLDER=1
  CHECKOUT_GH_PAGES=0
elif [ ${ACT} == "build_version" ]; then
  UPDATE_LATEST_FOLDER=0
  UPDATE_VERSION_FOLDER=1
  CHECKOUT_GH_PAGES=1
elif [ ${ACT} == "build_latest" ]; then
  UPDATE_LATEST_FOLDER=1
  UPDATE_VERSION_FOLDER=0
  CHECKOUT_GH_PAGES=1
fi

WORK_DIR=../../build_tmp
rm -rf /tmp/env_sphinx
if [ ! -d ${WORK_DIR} ]; then
  echo "no ${WORK_DIR}"

else
  if [ ! -d ${WORK_DIR}/env_sphinx ]; then
    echo "no exist ${WORK_DIR}/env_sphinx"
  else
    cp -rf ${WORK_DIR}/env_sphinx /tmp/
    rm -rf ${WORK_DIR}
    echo "backup ${WORK_DIR}/env_sphinx to /tmp"
  fi
fi

mkdir -p ${WORK_DIR}
cp -rf ./* ${WORK_DIR}

cd ${WORK_DIR}

if [ ! -d /tmp/env_sphinx ]; then
  echo "no /tmp/env_sphinx"
else
  echo "restore env_sphinx from /tmp"
  cp -r /tmp/env_sphinx ./
fi

if [ ! -d env_sphinx ]; then
  echo "create env_sphinx"
  bash setup_env.sh
fi

source env_sphinx/bin/activate

cp -rf ../docs/ ./source
cp -rf ../intel_extension_for_transformers ./source/docs
rm -rf ./source/build_docs
#cp -f "../README.md" "./source/docs/Welcome.md"
cp -f "../SECURITY.md" "./source/docs/SECURITY.md"


sed -i 's/docs\/imgs/.\/imgs/g' ./source/docs/Welcome.md
sed -i 's/.md/.html/g; s/.\/docs\/source\//.\/docs/g' ./source/docs/Welcome.md
sed -i 's/href=\"docs\//href=\"/g' ./source/docs/Welcome.md
sed -i 's/(docs\//(/g' ./source/docs/Welcome.md

echo "run doxygen to create document for C++ code"
rm -rf _build_doxygen
doxygen Doxyfile.in
echo "doxygen is done!"

make clean
make html

if [[ $? -eq 0 ]]; then
  echo "Sphinx build online documents successfully!"
else
  echo "Sphinx build online documents fault!"
  exit 1
fi

DRAFT_FOLDER=./draft
mkdir -p ${DRAFT_FOLDER}
VERSION=`cat source/version.txt`
DST_FOLDER=${DRAFT_FOLDER}/${VERSION}
LATEST_FOLDER=${DRAFT_FOLDER}/latest
SRC_FOLDER=build/html

RELEASE_FOLDER=./gh-pages
ROOT_DST_FOLDER=${RELEASE_FOLDER}/${VERSION}
ROOT_LATEST_FOLDER=${RELEASE_FOLDER}/latest

if [[ ${UPDATE_VERSION_FOLDER} -eq 1 ]]; then
  echo "create ${DST_FOLDER}"
  rm -rf ${DST_FOLDER}/*
  mkdir -p ${DST_FOLDER}
  cp -r ${SRC_FOLDER}/* ${DST_FOLDER}
  python update_html.py ${DST_FOLDER} ${VERSION}
  cp -r ./source/docs/imgs ${DST_FOLDER}/docs
  cp -r ./source/docs/intel_extension_for_transformers/backends/neural_engine/docs/imgs ${DST_FOLDER}/docs/intel_extension_for_transformers/backends/neural_engine/docs
  cp -r ./source/docs/intel_extension_for_transformers/backends/neural_engine/kernels/docs/imgs ${DST_FOLDER}/docs/intel_extension_for_transformers/backends/neural_engine/kernels/docs
  cp source/_static/index.html ${DST_FOLDER}
else
  echo "skip to create ${DST_FOLDER}"
fi

if [[ ${UPDATE_LATEST_FOLDER} -eq 1 ]]; then
  echo "create ${LATEST_FOLDER}"
  rm -rf ${LATEST_FOLDER}/*
  mkdir -p ${LATEST_FOLDER}
  cp -r ${SRC_FOLDER}/* ${LATEST_FOLDER}
  python update_html.py ${LATEST_FOLDER} ${VERSION}
  cp -r ./source/docs/imgs ${LATEST_FOLDER}/docs
  cp -r ./source/docs/intel_extension_for_transformers/backends/neural_engine/docs/imgs ${LATEST_FOLDER}/docs/intel_extension_for_transformers/backends/neural_engine/docs
  cp -r ./source/docs/intel_extension_for_transformers/backends/neural_engine/kernels/docs/imgs ${LATEST_FOLDER}/docs/intel_extension_for_transformers/backends/neural_engine/kernels/docs
  cp source/_static/index.html ${LATEST_FOLDER}
else
  echo "skip to create ${LATEST_FOLDER}"
fi

echo "Create document is done"

if [[ ${CHECKOUT_GH_PAGES} -eq 1 ]]; then
  GITHUB_URL=`git config --get remote.origin.url`
  echo "git clone ${GITHUB_URL}"
  git clone -b gh-pages --single-branch ${GITHUB_URL} ${RELEASE_FOLDER}
 
  if [[ ${UPDATE_VERSION_FOLDER} -eq 1 ]]; then
    python update_version.py ${ROOT_DST_FOLDER} ${VERSION}
    cp -rf ${DST_FOLDER} ${RELEASE_FOLDER}
  fi

  if [[ ${UPDATE_LATEST_FOLDER} -eq 1 ]]; then
    cp -rf ${LATEST_FOLDER} ${RELEASE_FOLDER}
  fi

else
  echo "skip pull gh-pages"
fi

echo "UPDATE_LATEST_FOLDER=${UPDATE_LATEST_FOLDER}"
echo "UPDATE_VERSION_FOLDER=${UPDATE_VERSION_FOLDER}"


if [[ $? -eq 0 ]]; then
  echo "create online documents successfully!"
else
  echo "create online documents fault!"
  exit 1
fi

exit 0
