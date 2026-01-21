
.. _building:

Building
========

P4P can be built in one of two ways.
As a python package, preferably managed by PIP.
As an EPICS module.

When built as a python package,
the `epicscorelibs <https://pypi.org/project/epicscorelibs>`_ and `pvxslibs <https://pypi.org/project/pvxslibs>`_ python packages will provide the EPICS Base and PVXS libraries.
When built as an EPICS module, ``epics-base`` and ``pvxs`` will be referenced through ``configure/RELEASE`` or ``configure/RELEASE.local``.
Details below...

Release tars available from https://github.com/epics-base/p4p/releases ,
although cloning from git tag name is suggested.

Dependencies
------------

Python packages:

* numpy >=1.6
* Cython >=0.29.32
* nose2 (Optional, recommended to run unittests)
* ply (Optional for core module, required for `gwpage`)
* Optional
  * `cothread <https://github.com/dls-controls/cothread>`_ needed by `p4p.client.cothread.Context`.
  * `qtpy <https://github.com/spyder-ide/qtpy>`_ needed for `p4p.client.Qt.Context`.

EPICS modules:

* EPICS Base >= 3.14.12
* PVXS >= 0.2.0

Python Package
--------------

Pre-built python wheels, and source tars, are published to `pypi.org <https://pypi.org/project/p4p>`_.
These are suggested for new users.
For example, installing into a newly created virtualenv: ::

    python -m virtualenv p4ptest
    . p4ptest/bin/activate
    python -m pip install -U pip
    python -m pip install p4p

Local (re)build as Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If it is desired to ignore the pre-built wheel files,
and to rebuild p4p from source in a virtualenv. ::

    python -m virtualenv p4ptest
    . p4ptest/bin/activate
    python -m pip install -U pip
    python -m pip install \
     --only-binary numpy \
     --no-binary epicscorelibs,pvxslibs,p4p \
     nose2 p4p
    python -m nose2 p4p   # Optional: runs automatic tests

Site where the installation environment does not have access to pypi.org may wish
to investigate ``pip download -h``.

Build as EPICS Module
---------------------

P4P can also be built as an EPICS Module,
with the additional python dependencies installed either through a system or python package manager.

.. note:: The ``$(PYTHON)`` make variable can be set to the full path of a ``python`` executable, or to a name found in ``$PATH``.

As a prerequisite, build the epics-base and `PVXS <https://epics-base.github.io/pvxs/building.html>`_ modules.
The paths to these will be referenced in ``configure/RELEASE.local`` below.

Install additional python dependencies.  eg. a Debian Linux host. ::

   sudo apt-get install python3-dev python3-numpy python3-ply python3-nose2 cython3

or with PIP ::

   python3 -m pip install numpy nose2 Cython ply

From from versioned source.  May replace "master" with release version number. ::

   git clone --branch master https://github.com/epics-base/p4p.git
   cd p4p

Set location of EPICS modules.  With EPICS >= 7.0.2::

   cat <<EOF > configure/RELEASE.local
   PVXS=/path/to/pvxs
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

    make PYTHON=python3.8
    make PYTHON=python3.8 clean
    make PYTHON=python3.14
    make PYTHON=python3.14 clean

.. note:: If ``PYTHON=`` is ever specified, then it must be specified for all targets except 'distclean'.

.. _builddeps:

Building EPICS dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the necessary EPICS modules are not present, then they may be built from source. ::

   sudo apt-get install libreadline6-dev libncurses5-dev perl
   git clone https://github.com/epics-base/epics-base.git
   git clone https://github.com/epics-base/pvxs.git
   cat <<EOF > pvxs/configure/RELEASE.local
   EPICS_BASE=$PWD/epics-base
   EOF
   cat <<EOF > p4p/configure/RELEASE.local
   PVXS=$PWD/pvxs
   EPICS_BASE=$PWD/epics-base
   EOF
   make -C epics-base
   make -C pvxs

CLI and unittests
~~~~~~~~~~~~~~~~~

To run the unittests: ::

   make nose

For testing purposes several simple command line client tools are provided.
For further information run: ::

   PYTHONPATH=$PWD/python3.8/linux-x86_64 python -m p4p.client.cli -h
