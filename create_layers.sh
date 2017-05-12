#!/usr/bin/env sh
set -e

CAFFE_HOME=/opt/caffe

sudo cp ./layers/*.cpp $CAFFE_HOME/src/caffe/layers/
sudo cp ./layers/*.cu $CAFFE_HOME/src/caffe/layers/
sudo cp ./layers/*.hpp $CAFFE_HOME/include/caffe/layers/
