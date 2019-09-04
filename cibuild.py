#!/usr/bin/env python
"""Windows .bat files suck, and I don't want to learn powershell just for this.
So hey, why not just script in python...
"""

from __future__ import print_function

import sys, os, platform
import shutil
import re
import subprocess as SP
import distutils.util
from glob import glob

# https://www.python.org/dev/peps/pep-0513/
# https://www.python.org/dev/peps/pep-0425/
# eg.
#  TAG-ABI-PLATFORM
#
#  cp27-cp27m-manylinux1_i686
#  cp27-cp27m-manylinux1_x86_64
#  cp35-none-win32
#  cp34-none-win_amd64
#  cp35-cp35m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64

def is_pre():
    with open('setup.py', 'r') as F:
        ver = re.match(r".*package_version\s*=\s*'([^']*)'.*", F.read(), flags=re.DOTALL).group(1)

    return ver.find('a')!=-1

if sys.version_info>=(3,7):
    # numpy only provides 3.7 wheels for recent releases,
    # and source builds seem to fail on travis?
    requirments = 'requirements-deb10.txt'

elif platform.system()=='Windows':
    # numpy wheels also not provided for windows until 1.11
    requirments = 'requirements-windows.txt'

else:
    # for maximum compatibility, build against old numpy
    requirments = 'requirements-deb8.txt'

os.environ['REFTEST_IGNORE_TRANSIENT'] = 'YES'

def call_py(args, **kws):
    print('EXEC', sys.executable, args, kws, 'in', os.getcwd())
    sys.stdout.flush()
    SP.check_call([sys.executable]+args, **kws)
    print('EXEC DONE', sys.executable, args, kws)

def docker(args):
    print("Available pythons")
    for py in glob("/opt/python/*/bin/python"):
        print("  ", py, py==sys.executable)
    print("Switch to /io")
    os.chdir('/io')
    SP.check_call('gcc --version', shell=True)
    SP.check_call('ld --version', shell=True)


def prepare(args):
    call_py(['-m', 'pip', 'install', '-U', 'pip'])
    call_py(['-m', 'pip', 'install', '-r', requirments])
    call_py(['-m', 'pip', 'install', '-U', 'wheel', 'setuptools', 'twine'])
    if is_pre():
        print('Install pre-release dependencies')
        call_py(['-m', 'pip', 'install', '-U', '--pre', 'setuptools_dso'])
        call_py(['-m', 'pip', 'install', '-U', '--pre', '--only-binary', ':all:', 'epicscorelibs'])
    else:
        print('Install release dependencies')
        call_py(['-m', 'pip', 'install', '-U', 'setuptools_dso'])
        call_py(['-m', 'pip', 'install', '-U', '--only-binary', ':all:', 'epicscorelibs'])

def build(args):
    tag = args.pop(0)
    print('ABI tag', tag)
    
    call_py(['setup.py', 'clean', '-a']) # serves to verify that ./setup.py exists before we delete anything

    shutil.rmtree('dist', ignore_errors=True)
    shutil.rmtree('build', ignore_errors=True)

    call_py(['setup.py', 'sdist'])
    call_py(['setup.py', '-v', 'bdist_wheel', '-p', tag])

    results = glob('dist/*.whl')
    print('RESULT', results)

    if len(results)!=1:
        print('Too many wheels?!?')
        sys.exit(1)

    call_py(['-m', 'pip', 'install', results[0]])
    # prevent overzealous nose from inspecting src/
    os.chdir('dist')
    nose = ['-m', 'nose', 'p4p', '-v']
    call_py(nose)
    os.chdir('..')

def upload(args):
    if 'APPVEYOR_PULL_REQUEST_NUMBER' in os.environ or 'TWINE_USERNAME' not in os.environ:
        print("APPVEYOR is PR, skip upload attempt")
        return

    files = []
    files.extend(glob('dist/*.whl'))
    files.extend(glob('dist/*.tar.*'))

    call_py(['-m', 'twine', 'upload', '--skip-existing']+files)

actions = {
    'docker': docker,
    'prepare': prepare,
    'build': build,
    'upload': upload,
}

if __name__=='__main__':
    print(sys.version)

    print('PYTHONPATH')
    for dname in sys.path:
        print(' ', dname)

    print('platform =', distutils.util.get_platform())

    try:
        from pip._internal import pep425tags
    except ImportError:
        print('No pip?')
    else:
        print('PIP compatible')
        for parts in pep425tags.get_supported():
            print('  ', "-".join(parts))

    try:
        from wheel import pep425tags
    except ImportError:
        print('No wheel?')
    else:
        print('Wheel compatible')
        for parts in pep425tags.get_supported():
            print('  ', "-".join(parts))

    args = sys.argv[1:]
    while len(args)>0:
        name = args.pop(0)
        print('IN', name, 'with', args)
        actions[name](args)
