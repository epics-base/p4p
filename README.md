PVAccess for Python
===================

Python bindings for the PVAccess network client and server.

    python -m virtualenv p4ptest
    . p4ptest/bin/activate
    python -m pip install -U pip
    python -m pip install p4p
    python -m nose2 p4p

The tests must be run from outside the p4p source tree.
In a git clone, `src/p4p` is found before the installed package and the
tests fail with `ImportError: cannot import name '_p4p'`.

See documentation

https://epics-base.github.io/p4p/

Source

https://github.com/epics-base/p4p/

CI results

![Github Actions](https://github.com/epics-base/p4p/workflows/P4P/badge.svg)
