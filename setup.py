#!/usr/bin/env python

from __future__ import print_function

from setuptools_dso import Extension, setup

import numpy
from numpy.distutils.misc_util import get_numpy_include_dirs

import epicscorelibs.path
import epicscorelibs.version
from epicscorelibs.config import get_config_var

extra = []
import sys
if sys.platform=='linux2':
    extra += ['-v']

ext = Extension(
    name='p4p._p4p',
    sources = [
        "src/p4p_top.cpp",
        "src/p4p_type.cpp",
        "src/p4p_value.cpp",
        "src/p4p_array.cpp",

        "src/p4p_server.cpp",
        "src/p4p_server_provider.cpp",
        "src/p4p_server_sharedpv.cpp",

        "src/p4p_client.cpp",
    ],
    include_dirs = get_numpy_include_dirs()+[epicscorelibs.path.include_path],
    define_macros = get_config_var('CPPFLAGS'),
    extra_compile_args = get_config_var('CXXFLAGS'),
    extra_link_args = get_config_var('LDFLAGS')+extra,
    dsos = ['epicscorelibs.lib.pvAccess',
            'epicscorelibs.lib.pvData',
            'epicscorelibs.lib.ca',
            'epicscorelibs.lib.Com'
    ],
    libraries = get_config_var('LDADD'),
)

setup(
    name='p4p',
    version='1.1a36',
    description="Python interface to PVAccess protocol client",
    url='https://mdavidsaver.github.io/p4p',
    author='Michael Davidsaver',
    author_email='mdavidsaver@gmail.com',
    license='EPICS',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'License :: Freely Distributable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: System :: Distributed Computing',
    ],
    keywords='epics scada',
    python_requires='>=2.7',

    packages=[
        'p4p',
        'p4p.nt',
        'p4p.client',
        'p4p.test',
        'p4p.server',
    ],
    package_dir={'':'src'},
    ext_modules = [ext],
    install_requires = [
        epicscorelibs.version.abi_requires(),
        # assume ABI forward compatibility as indicated by
        # https://github.com/numpy/numpy/blob/master/numpy/core/setup_common.py#L28
        'numpy >=%s'%numpy.version.short_version,
        'nose>=1.1.2',
    ],
    zip_safe = False,
)
