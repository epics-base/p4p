#!/usr/bin/env python

from __future__ import print_function

import sysconfig

from setuptools_dso import Extension, setup, cythonize

import numpy
from numpy.distutils.misc_util import get_numpy_include_dirs

import epicscorelibs.path
import epicscorelibs.version
from epicscorelibs.config import get_config_var

# the following line is matched from cibuild.py
package_version = '3.5.3'

cxxflags = []
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
    extra_compile_args = get_config_var('CXXFLAGS')+cxxflags,
    extra_link_args = get_config_var('LDFLAGS')+ldflags,
    dsos = ['epicscorelibs.lib.pvAccess',
            'epicscorelibs.lib.pvData',
            'epicscorelibs.lib.ca',
            'epicscorelibs.lib.Com'
    ],
    libraries = get_config_var('LDADD'),
)

gwext = cythonize([
    Extension(
        name='p4p._gw',
        sources=['src/p4p/_gw.pyx', 'src/gwchannel.cpp'],
        include_dirs = get_numpy_include_dirs()+[epicscorelibs.path.include_path, 'src', 'src/p4p'],
        define_macros = get_config_var('CPPFLAGS'),
        extra_compile_args = get_config_var('CXXFLAGS')+cxxflags,
        extra_link_args = get_config_var('LDFLAGS')+ldflags,
        dsos = ['epicscorelibs.lib.pvAccess',
            'epicscorelibs.lib.pvData',
            'epicscorelibs.lib.ca',
            'epicscorelibs.lib.Com'
        ],
        libraries = get_config_var('LDADD'),
    )
])

setup(
    name='p4p',
    version=package_version,
    description="Python interface to PVAccess protocol client",
    url='https://mdavidsaver.github.io/p4p',
    author='Michael Davidsaver',
    author_email='mdavidsaver@gmail.com',
    license='BSD',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'License :: Freely Distributable',
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
    ext_modules = [ext]+gwext,
    install_requires = [
        epicscorelibs.version.abi_requires(),
        # assume ABI forward compatibility as indicated by
        # https://github.com/numpy/numpy/blob/master/numpy/core/setup_common.py#L28
        'numpy >=%s'%numpy.version.short_version,
        'nose>=1.1.2',
        'ply', # for asLib
    ],
    extras_require={
        'qt': ['qtpy'],
    },
    entry_points = {
        'console_scripts': ['pvagw=p4p.gw:main'],
    },
    zip_safe = False,
)
