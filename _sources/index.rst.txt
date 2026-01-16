PVAccess for Python (P4P)
=========================

**P4P** is a wrapper around the PVAccess (aka. PVA) protocol network client and server.
P4P is part of the **EPICS** (Experimental Physics and Industrial Control System) ecosystem
for creating large (and small) scale distributed process control and data acqisition sytems.
https://epics.anl.gov/

P4P includes API for a `clientapi` and `serverapi`,
as well as a `gwpage` executable (``pvagw``) which can optionally enforce an access control policy.

Supports Linux, OSX, and Windows.  With python 2.7 and >=3.5 (>=3.6 for asyncio support).

The recommended starting point is to install from pypi.org. ::

    python -m virtualenv p4ptest
    . p4ptest/bin/activate
    python -m pip install -U pip
    python -m pip install p4p nose2
    python -m nose2 p4p   # Optional: runs automatic tests

- VCS: https://github.com/epics-base/p4p
- Docs: https://epics-base.github.io/p4p/

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

