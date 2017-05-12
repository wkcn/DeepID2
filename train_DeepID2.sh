#!/usr/bin/env sh
set -e

TOOLS=/opt/caffe/build/tools

$TOOLS/caffe train --solver=./DeepID2_solver.prototxt $@
