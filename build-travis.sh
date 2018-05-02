#!/bin/sh
set -e -x

PLATNAME=$1

cd /io

ls /opt/python/*/bin/python
install -d wheels

# hack for py3 specific tests
mv src/p4p/test/test_asyncio.py save

for PYBIN in /opt/python/*/bin; do
    cd /io

    if "${PYBIN}/python" -c 'import sys; sys.exit(not (sys.version_info.major>=3))'
    then
        cp save src/p4p/test/test_asyncio.py
    fi

    # skip if deps fail to install (numpy, I'm looking at you!)
    "${PYBIN}/pip" install --only-binary all -r requirements-deb9.txt || continue
    # skip if no wheel for our primary dep
    "${PYBIN}/pip" install --only-binary all epicscorelibs || continue
    # delete all temp.  with py2.7 this avoids mixing object code from cp27m and cp27mu
    "${PYBIN}/python" setup.py clean -a
    "${PYBIN}/python" setup.py sdist
    "${PYBIN}/python" setup.py bdist_wheel -v -p "$PLATNAME"

    cd dist

    "${PYBIN}/pip" install *.whl

    echo "PYTHONPATH=$PYTHONPATH"
    "${PYBIN}/python" -m nose p4p

    cd ..

    mv dist/*.whl wheels/
    mv dist/*.tar.* wheels/
done
