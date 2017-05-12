#!/usr/bin/env sh
set -e

CAFFE_HOME=/opt/caffe

cp ./layers/*.cpp $CAFFE_HOME/src/caffe/layers/
cp ./layers/*.cu $CAFFE_HOME/src/caffe/layers/
cp ./layers/*.hpp $CAFFE_HOME/include/caffe/layers/
