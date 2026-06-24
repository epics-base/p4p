#!/usr/bin/env python
"""
Set some python derived Makefile variables.

Emits something like the following

PY_OK := YES  # indicates success of this script
PY_VER := 2.6
PY_INCDIRS := /path ...
PY_LIBDIRS := /path ...
"""

from __future__ import print_function

import sys
import os

if len(sys.argv)<2:
    out = sys.stdout
else:
    try:
        os.makedirs(os.path.dirname(sys.argv[1]))
    except OSError:
        pass
    out = open(sys.argv[1], 'w')

try:
    from sysconfig import get_config_var, get_path
    def get_python_inc():
        return get_path('include')
except ImportError:
    from distutils.sysconfig import get_config_var, get_python_inc

def gcv(name, *dflt):
    v = get_config_var(name)
    if v is None:
        if len(dflt):
            return dflt[0]
        raise KeyError(name)
    return v

incdirs = [get_python_inc()]
libdir = gcv('LIBDIR', '') or gcv('prefix') + '/libs'


def get_numpy_include_dirs():
    from numpy import get_include

    return [get_include()]


incdirs = get_numpy_include_dirs() + incdirs

target_flags = gcv('BASECFLAGS', '')
print('TARGET_CFLAGS +=',target_flags, file=out)
print('TARGET_CXXFLAGS +=',target_flags, file=out)

print('PY_VER :=',gcv('VERSION'), file=out)
ldver = gcv('LDVERSION', None)
if ldver is None:
    ldver = gcv('VERSION')
    if gcv('Py_DEBUG', ''):
        ldver = ldver+'_d'
print('PY_LD_VER :=',ldver, file=out)
print('PY_INCDIRS :=',' '.join(incdirs), file=out)
print('PY_LIBDIRS :=',libdir, file=out)

try:
    import asyncio
except ImportError:
    print('HAVE_ASYNCIO := NO', file=out)
else:
    print('HAVE_ASYNCIO := YES', file=out)

try:
    import cothread
except ImportError:
    print('HAVE_COTHREAD := NO', file=out)
else:
    print('HAVE_COTHREAD := YES', file=out)

print('PY_OK := YES', file=out)

out.close()
