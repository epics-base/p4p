#!/usr/bin/env python
"""
Set some python derived Makefile variables.

Emits something like the following

PY_OK := YES  # indicates success of this script
HAVE_NUMPY := YES/NO
PY_VER := 2.6
PY_INCDIRS := /path ...
PY_LIBDIRS := /path ...
"""

from __future__ import print_function

import sys

if len(sys.argv)<2:
    out = sys.stdout
else:
    out = open(sys.argv[1], 'w')

from distutils.sysconfig import get_config_var, get_python_inc

incdirs = [get_python_inc()]
libdirs = [get_config_var('LIBDIR')]

have_np='NO'
try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
    incdirs = get_numpy_include_dirs()+incdirs
    have_np='YES'
except ImportError:
    pass

print('TARGET_CFLAGS +=',get_config_var('BASECFLAGS'), file=out)
print('TARGET_CXXFLAGS +=',get_config_var('BASECFLAGS'), file=out)

print('PY_VER :=',get_config_var('VERSION'), file=out)
ldver = get_config_var('LDVERSION')
if ldver is None:
    ldver = get_config_var('VERSION')
    if get_config_var('Py_DEBUG'):
        ldver = ldver+'_d'
print('PY_LD_VER :=',ldver, file=out)
print('PY_INCDIRS :=',' '.join(incdirs), file=out)
print('PY_LIBDIRS :=',' '.join(libdirs), file=out)
print('HAVE_NUMPY :=',have_np, file=out)

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
