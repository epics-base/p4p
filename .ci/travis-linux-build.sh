#!/bin/sh
set -e -x

make PYTHON=`which python` -j2

make PYTHON=`which python` nose
