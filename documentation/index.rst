PVAccess for Python (P4P)
=========================

The canonical version of this page is https://mdavidsaver.github.io/p4p/

**P4P** is a wrapper around the PVAccess (aka. PVA) protocol network client and server.
P4P is part of the **EPICS** (Experimental Physics and Industrial Control System) ecosystem
for creating large (and small) scale distributed process control and data acqisition sytems.
https://epics.anl.gov/

P4P includes API for a `clientapi` and `serverapi`,
as well as a `gwpage` executable (``pvagw``) with which can enforce an access control policy.

Supports Linux, OSX, and Windows.  With python 2.7 and >=3.4.

The recommended starting point is to install from pypi.org. ::

    python -m virtualenv p4ptest
    . p4ptest/bin/activate
    python -m pip install -U pip
    python -m pip install p4p
    python -m nose p4p   # Optional: runs automatic tests

Release tars can be downloaded from https://github.com/mdavidsaver/p4p/releases

Versioned source can be found at https://github.com/mdavidsaver/p4p

Contents:

.. toctree::
   :maxdepth: 2

   starting
   overview
   building
   client
   nt
   values
   server
   rpc
   internal
   gw

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

