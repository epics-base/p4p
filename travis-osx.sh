#!/bin/sh
set -e -x

gcc --version
clang --version

# https://github.com/joerick/cibuildwheel/blob/master/cibuildwheel/macos.py

curl -L -o /tmp/Python.pkg $URL
curl -L -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py

sudo installer -pkg /tmp/Python.pkg -target /

export PATH=$PATH:/Library/Frameworks/Python.framework/Versions/$PYVER/bin

which $PYTHON

$PYTHON --version

$PYTHON /tmp/get-pip.py --no-setuptools --no-wheel

$PYTHON -m pip --version
