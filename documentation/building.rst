
.. _building:

Building
========

Release tars available from https://github.com/mdavidsaver/p4p/releases

The P4P modules requires:

* Python 2.7 or >=3.4
* numpy >=1.6
* nosetests (to run unittests)

* EPICS >= 7.0.2

or

* EPICS Base >= 3.14.12
* pvDataCPP >=7.1.0
* pvAccessCPP >=6.1.0

P4P can be built and installed in one of two ways.
As a python package, preferably managed by PIP.
As an EPICS module.

Build as Python package
-----------------------

The process for building as a python package using sources from pypi.org. ::

    python -m virtualenv p4ptest
    . p4ptest/bin/activate
    python -m pip install -U pip
    python -m pip install nose numpy # Optional: avoids building numpy from source (slow)
    python -m pip install --no-binary :all: p4p
    python -m nose p4p   # Optional: runs automatic tests

Build as EPICS Module
---------------------

Install python dependencies on a Debian Linux host::

   sudo apt-get install python2.7-dev python-numpy python-nose

or with PIP::

   pip install -r requirements-deb9.txt

From release tar.::

   curl -L 'https://github.com/mdavidsaver/p4p/releases/download/1.0/p4p-1.0.tar.gz' | tar -xz
   cd p4p-1.0

or from from versioned source.::

   git clone https://github.com/mdavidsaver/p4p.git
   cd p4p

Set location of EPICS modules.  With EPICS >= 7.0.1::

   cat <<EOF > configure/RELEASE.local
   EPICS_BASE=/path/to/epics-base
   EOF
   make

See below for details on building EPICS from source.

By default P4P will build using 'python' found in the system search path.
To explicitly specify a particular version. ::

   make distclean
   make PYTHON=python3.4

Alternately, the full path of the interpreter can be used. ::

   make distclean
   make PYTHON=/usr/bin/python3.4

For convenience PYTHON can also be set in configure/CONFIG_SITE

Multiple Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~

To build for multiple python versions it is necessary to do a partial clean before building
another version.  This will not remove the final tree. ::

    make PYTHON=python2.7
    make PYTHON=python2.7 clean
    make PYTHON=python3.4
    make PYTHON=python3,4 clean

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

   PYTHONPATH=$PWD/python2.7/linux-x86_86 nosetests

For testing purposes several simple command line client tools are provided.
For further information run: ::

   PYTHONPATH=$PWD/python2.7/linux-x86_86 python -m p4p.client.cli -h
