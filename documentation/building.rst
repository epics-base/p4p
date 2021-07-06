
.. _building:

Building
========

Release tars available from https://github.com/mdavidsaver/p4p/releases

The P4P modules requires:

* Python 2.7 or >=3.4
* numpy >=1.6
* Cython >=0.25.2
* nosetests (Optional, recommended to run unittests)

and

* EPICS >= 7.0.2

or

* EPICS Base >= 3.14.12
* pvDataCPP >=7.1.0
* pvAccessCPP >=6.1.0

P4P can be built and installed in one of two ways.
As a python package, preferably managed by PIP.
As an EPICS module.

Optional

* `cothread <https://github.com/dls-controls/cothread>`_ needed by `p4p.client.cothread.Context`.
* `qtpy <https://github.com/spyder-ide/qtpy>`_ needed for `p4p.client.Qt.Context`.

Local (re)build as Python package
---------------------------------

The process for building as a python package using sources from pypi.org by adding the argument "--no-binary"
to prevent the use of pre-built binarys (except for numpy). ::

    python -m virtualenv p4ptest
    . p4ptest/bin/activate
    python -m pip install -U pip
    python -m pip install \
     --only-binary numpy \
     --no-binary epicscorelibs,p4p \
     p4p
    python -m nose p4p   # Optional: runs automatic tests

Bootstrap a virtualenv offline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It may be necessary to create a virtualenv including P4P on a machine with no internet access,
or to ensure the use of certain verified binary/source.
The following recipe was tested with virtualenv 16.1 and pip 18.1.

First, from a machine with internet access, and having the same archetecture as the target machine,
collect the necessary files. ::

    mkdir /tmp/venv
    cd /tmp/venv
    python -m pip download virtualenv
    unzip virtualenv-*.whl
    python virtualenv.py --never-download prepenv
    . prepenv/bin/activate
    pip download -d virtualenv_support p4p
    tar -caf p4p-env-`date -u +%Y%m%d`.tar.gz virtualenv.py virtualenv_support
    deactivate

Now move the created file p4p-env-\*.tar.gz to the target machine.
Then prepare the virtualenv env with. ::

    tar -xaf p4p-env-*.tar.gz
    python virtualenv.py --never-download env
    . env/bin/activate
    pip install --no-index -f virtualenv_support p4p
    python -m nose p4p   # Optional: runs automatic tests

Utilities to automate this process include https://pypi.org/project/pyutilib.virtualenv/

Build as EPICS Module
---------------------

P4P can also be built as an EPICS Module with additional python dependencies.

Prepare the host to build python extensions.  eg. a Debian Linux host::

   sudo apt-get install build-essential python3-dev

Install python dependencies on eg. a Debian Linux host::

   sudo apt-get install python3-numpy python3-nose cython3

or with PIP::

   python3 -m pip install -r requirements-latest.txt

From release tar.::

   curl -L 'https://github.com/mdavidsaver/p4p/releases/download/1.0/p4p-3.5.4.tar.gz' | tar -xz
   cd p4p-3.5.4

or from from versioned source.::

   git clone https://github.com/mdavidsaver/p4p.git
   cd p4p

Set location of EPICS modules.  With EPICS >= 7.0.2::

   cat <<EOF > configure/RELEASE.local
   EPICS_BASE=/path/to/epics-base
   EOF
   make

See below for details on building EPICS from source.

By default P4P will build using 'python' found in the system search path.
To explicitly specify a particular version. ::

   make distclean
   make PYTHON=python3

Alternately, the full path of the interpreter can be used. ::

   make distclean
   make PYTHON=/usr/bin/python3

For convenience PYTHON can also be set in configure/CONFIG_SITE

Multiple Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~

To build for multiple python versions it is necessary to do a partial clean before building
another version.  This will not remove the final tree. ::

    make PYTHON=python2
    make PYTHON=python2 clean
    make PYTHON=python3
    make PYTHON=python3 clean

.. note:: If PYTHON= is ever specified, then it must be specified for all targets except 'distclean'.

.. _builddeps:

Building EPICS dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the necessary EPICS modules are not present, then they may be built from source. ::

   sudo apt-get install libreadline6-dev libncurses5-dev perl
   git clone --recursive https://github.com/epics-base/epics-base.git
   make -C epics-base
   echo "EPICS_BASE=$PWD/epics-base" > ../p4p/configure/RELEASE.local

When building against EPICS < 7.0.1, the pvDataCPP and pvAccessCPP modules
must be built separately.

CLI and unittests
~~~~~~~~~~~~~~~~~

To run the unittests: ::

   make nose

or (change path as appropriate)::

   PYTHONPATH=$PWD/python2.7/linux-x86_64 nosetests

For testing purposes several simple command line client tools are provided.
For further information run: ::

   PYTHONPATH=$PWD/python2.7/linux-x86_64 python -m p4p.client.cli -h
