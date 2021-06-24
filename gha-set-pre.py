#!/usr/bin/env python
"""Push PRE=--pre
to the GHA environment for subsequent actions if building a pre-release.
"""

from __future__ import print_function

import os
import re

with open('setup.py', 'r') as F:
    ver = re.match(r".*package_version\s*=\s*'([^']*)'.*", F.read(), flags=re.DOTALL).group(1)

if ver.find('a')!=-1:
    print('Is pre-release')
    # https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions#setting-an-environment-variable
    #echo "{name}={value}" >> $GITHUB_ENV

    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a') as F:
            F.write('PRE=--pre\n')
    else:
        print('Would export PRE=--pre')
