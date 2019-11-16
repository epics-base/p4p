#!/bin/sh
set -e -x

/sbin/ifconfig -a

python -m pip install -r requirements-${PROF}.txt

python -m pip install cothread

./.ci/build-deps.sh "$@"
