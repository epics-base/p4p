Building
========

The P4P modules requires:

* Python 2.7 or >=3.4
* numpy >=1.6
* nosetests (to run unittests)
* EPICS Base >= 3.14.12
* PVDataCPP (unreleased)
* PVAccessCPP (unreleased)

Build from source
-----------------

Fetch the source.::

   git clone https://github.com/mdavidsaver/p4p.git
   cd p4p

Setup on a Debian Linux host::

   sudo apt-get install python2.7-dev python-numpy nose

or with PIP::

   pip install -r requirements-deb8.txt
   

Set location of EPICS modules::

   cat <<EOF > configure/RELEASE.local
   PVACCESS=/path/to/pvAccessCPP
   PVDATA=/path/to/pvDataCPP
   EPICS_BASE=/path/to/epics-base
   EOF
   make

By default P4P will build using 'python' found in the system search path.
To explicitly specify a particular version.::

   make distclean
   make PYTHON=python3.4

Alternately, the full path of the interpreter can be used. ::

   make distclean
   make PYTHON=/usr/bin/python3.4

For convenience PYTHON can also be set in configure/CONFIG_SITE

Multiple Python Versions
------------------------

To build for multiple python versions it is necessary to do a partial clean before building
another version.  This will not remove the final tree. ::

    make PYTHON=python2.7
    make PYTHON=python2.7 clean
    make PYTHON=python3.4
    make PYTHON=python3,4 clean

.. _builddeps:

Building EPICS dependencies
---------------------------

If the necessary EPICS modules are not present, then they may be built form source.
Note that the 'pva2pva' module is not required to build P4P, and may be omitted.
It is used in the :ref:`starting` demo. ::

   git clone --recursive https://github.com/epics-base/epics-base.git
   git clone https://github.com/epics-base/pvDataCPP.git
   git clone https://github.com/epics-base/pvAccessCPP.git
   git clone https://github.com/epics-base/pva2pva.git
   cat <<EOF >  RELEASE.local
   PVACCESS=$PWD/pvAccessCPP
   PVDATA=$PWD/pvDataCPP
   EPICS_BASE=$PWD/epics-base
   EOF
   make -C epics-base -j2
   make -C pvDataCPP -j2
   make -C pvAccessCPP -j2
   make -C pva2pva -j2


CLI and unittests
-----------------

To run the unittests: ::

   make nose

or (change path as approriate)::

   PYTHONPATH=$PWD/python2.7/linux-x86_86 nosetests

For testing purposes several simple command line client tools are provided.
For further information run: ::

   PYTHONPATH=$PWD/python2.7/linux-x86_86 python -m p4p.client.cli -h
