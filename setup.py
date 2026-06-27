#!/usr/bin/env python

from __future__ import print_function

import os
import shutil
import subprocess
import sysconfig

from setuptools import Command
from setuptools_dso import Extension, setup, cythonize, build_ext as _build_ext

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
if sys.platform.startswith('linux') and not sysconfig.get_config_var('Py_DEBUG'):
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

# Fully-qualified module names for which .pyi stubs are generated.
# Must match the Extension name= values below.
_STUB_MODULES = ['p4p._p4p', 'p4p._gw']


class generate_stubs(Command):
    """Generate .pyi stub files for compiled extensions using mypy stubgen.

    Invoked automatically by build_ext. To regenerate stubs without
    rebuilding the extensions::

        python setup.py generate_stubs
    """

    description = 'generate .pyi stub files for compiled extensions'
    user_options = []

    def initialize_options(self): pass
    def finalize_options(self): pass

    def _find_stubgen(self):
        """Locate the stubgen console script installed alongside this Python.

        The scripts directory is checked first so that the environment-local
        stubgen is preferred over any system-wide installation.
        """
        stubgen = shutil.which('stubgen', path=sysconfig.get_path('scripts')) \
            or shutil.which('stubgen')

        if stubgen is None:
            print(
                "WARNING: stubgen not found; skipping stub generation. "
                "Typing stub files (*.pyi) will not be generated",
                file=sys.stderr,
            )
            return None
        
        return stubgen

    def _make_env(self):
        """Return a copy of os.environ with pvxslibs/epicscorelibs lib dirs
        prepended to the platform DSO search path variable, so that stubgen
        subprocesses can load the extensions in editable installs where the
        relative RUNPATH embedded in the .so/.pyd does not resolve correctly.
        """
        # pvxslibs exposes no public lib_path attribute; path derived by convention.
        lib_dirs = [d for d in [
            os.path.join(os.path.dirname(pvxslibs.__file__), 'lib'),
            epicscorelibs.path.lib_path,
        ] if os.path.isdir(d)]
        env = os.environ.copy()
        if sys.platform == 'win32':
            path_var = 'PATH'
        elif sys.platform == 'darwin':
            path_var = 'DYLD_LIBRARY_PATH'
        else:
            path_var = 'LD_LIBRARY_PATH'
        existing = env.get(path_var, '')
        env[path_var] = os.pathsep.join(lib_dirs + ([existing] if existing else []))

        return env

    def run(self):
        """Verify extensions are importable, then invoke stubgen to write .pyi files."""
        stubgen = self._find_stubgen()
        if stubgen is None:
            return
        env = self._make_env()

        # For non-inplace builds (e.g. wheel), extensions land in build_lib rather
        # than the source tree, so add it to PYTHONPATH for the subprocesses.
        build_ext_cmd = self.get_finalized_command('build_ext')
        if not build_ext_cmd.inplace:
            existing = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = os.pathsep.join(
                [build_ext_cmd.build_lib] + ([existing] if existing else [])
            )

        # Verify extensions are importable under the same env stubgen will use.
        # Done in a subprocess so the current process's DSO search path (which
        # may not include pvxslibs/lib for editable installs) is not a factor.
        for mod in _STUB_MODULES:
            check = subprocess.run(
                [sys.executable, '-c', 'import %s' % mod],
                env=env, capture_output=True, check=False,
            )
            if check.returncode:
                print(
                    "WARNING: extension %r not importable; skipping stub generation. "
                    "Build it first with:\n"
                    "  python setup.py build_ext --inplace" % mod,
                    file=sys.stderr,
                )
                return

        # Run stubgen in a fresh subprocess so that Windows multiprocessing
        # (which stubgen uses internally to import extension modules safely)
        # works without needing a 'if __name__ == "__main__":' guard here.
        cmd = [stubgen, '--include-docstrings', '--output', 'src']
        for mod in _STUB_MODULES:
            cmd += ['-m', mod]

        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode:
            print("WARNING: stubgen failed with code %s" % result.returncode, file=sys.stderr)
            return

        missing = [
            os.path.join('src', mod.replace('.', os.sep) + '.pyi')
            for mod in _STUB_MODULES
            if not os.path.isfile(os.path.join('src', mod.replace('.', os.sep) + '.pyi'))
        ]
        if missing:
            print(
                "WARNING: expected stubs not found:\n" + '\n'.join(missing),
                file=sys.stderr,
            )


class build_ext(_build_ext):
    def run(self):
        super(build_ext, self).run()
        self.run_command('generate_stubs')


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
    package_data={'p4p': ['*.conf', '*.service', '*.pyi']},
    cmdclass={'build_ext': build_ext, 'generate_stubs': generate_stubs},
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
