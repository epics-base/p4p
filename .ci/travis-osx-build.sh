#!/bin/sh
set -e -x

make PYTHON=`which $PYTHON` -j2

make PYTHON=`which $PYTHON` nose
