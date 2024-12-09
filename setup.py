#!/usr/bin/env python

from __future__ import print_function

import os
import sysconfig

from setuptools_dso import Extension, setup, cythonize

import numpy

import epicscorelibs.path
import epicscorelibs.version
from epicscorelibs.config import get_config_var

import pvxslibs.path
import pvxslibs.version


def get_numpy_include_dirs():
    return [numpy.get_include()]


with open('src/p4p/version.py', 'r') as F:
    lcl = {}
    exec(F.read(), None, lcl)
    package_version = str(lcl['version'])
    del lcl

cxxflags = []
if get_config_var('CMPLR_CLASS') in ('gcc', 'clang'):
    cxxflags += ['-std=c++11']
ldflags = []
import sys
import platform
if sys.platform=='linux2' and not sysconfig.get_config_var('Py_DEBUG'):
    # c++ debug symbols size is huge.  ~20x code size.
    # So we choose to only emit debug symbols when building for an interpreter
    # with debugging enabled (aka 'python-dbg' on debian).
    cxxflags += ['-g0']

elif platform.system()=='Darwin':
    # avoid later failure where install_name_tool may run out of space.
    #   install_name_tool: changing install names or rpaths can't be redone for:
    #   ... because larger updated load commands do not fit (the program must be relinked,
    #   and you may need to use -headerpad or -headerpad_max_install_names)
    ldflags += ['-Wl,-headerpad_max_install_names']

# Our internal interfaces with generated cython
# are all c++, and MSVC doesn't allow extern "C" to
# return c++ types.
cppflags = get_config_var('CPPFLAGS') + [('__PYX_EXTERN_C','extern')]

exts = cythonize([
    Extension(
        name='p4p._p4p',
        sources = [
            "src/p4p/_p4p.pyx",
            "src/pvxs_client.cpp",
            "src/pvxs_sharedpv.cpp",
            "src/pvxs_source.cpp",
            "src/pvxs_type.cpp",
            "src/pvxs_value.cpp",
        ],
        include_dirs = get_numpy_include_dirs()+[epicscorelibs.path.include_path, pvxslibs.path.include_path, 'src', 'src/p4p'],
        define_macros = cppflags + [
            ('PY_ARRAY_UNIQUE_SYMBOL', 'PVXS_PyArray_API'),
            ('PVXS_ENABLE_EXPERT_API', None),
        ],
        extra_compile_args = get_config_var('CXXFLAGS')+cxxflags,
        extra_link_args = get_config_var('LDFLAGS')+ldflags,
        dsos = ['pvxslibs.lib.pvxs',
                'epicscorelibs.lib.Com'
        ],
        libraries = get_config_var('LDADD'),
    ),
    Extension(
        name='p4p._gw',
        sources=[
            'src/p4p/_gw.pyx',
            'src/pvxs_gw.cpp',
            'src/pvxs_odometer.cpp'
        ],
        include_dirs = get_numpy_include_dirs()+[epicscorelibs.path.include_path, pvxslibs.path.include_path, 'src', 'src/p4p'],
        define_macros = cppflags + [('PVXS_ENABLE_EXPERT_API', None)],
        extra_compile_args = get_config_var('CXXFLAGS')+cxxflags,
        extra_link_args = get_config_var('LDFLAGS')+ldflags,
        dsos = ['pvxslibs.lib.pvxs',
                'epicscorelibs.lib.Com'
        ],
        libraries = get_config_var('LDADD'),
    )
])

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as F:
    long_description = F.read()

install_requires = [
    epicscorelibs.version.abi_requires(),
    pvxslibs.version.abi_requires(),
    'nose2>=0.8.0',
    'ply', # for asLib
]

if hasattr(numpy.lib, "NumpyVersion") and numpy.lib.NumpyVersion(numpy.__version__) >= '2.0.0b1':
    install_requires += ['numpy >= 1.7', 'numpy < 3']
else:
    # assume ABI forward compatibility as indicated by
    # https://github.com/numpy/numpy/blob/master/numpy/core/setup_common.py#L28
    install_requires += ['numpy >=%s'%numpy.version.short_version, 'numpy < 2']

setup(
    name='p4p',
    version=package_version,
    description="Python interface to PVAccess protocol client",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://epics-base.github.io/p4p',
    author='Michael Davidsaver',
    author_email='mdavidsaver@gmail.com',
    license='BSD',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: System :: Distributed Computing',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
    ],
    keywords='epics scada',
    python_requires='>=2.7',

    packages=[
        'p4p',
        'p4p.nt',
        'p4p.client',
        'p4p.test',
        'p4p.server',
        'p4p.asLib',
    ],
    package_dir={'':'src'},
    package_data={'p4p': ['*.conf', '*.service']},
    ext_modules = exts,
    install_requires = install_requires,
    extras_require={
        'qt': ['qtpy'],
    },
    entry_points = {
        'console_scripts': ['pvagw=p4p.gw:main'],
    },
    zip_safe = False,
)
