#!/usr/bin/env python
"""Push PRE=--pre
to the GHA environment for subsequent actions if building a pre-release.
"""

import os

with open('src/p4p/version.py', 'r') as F:
    lcl = {}
    exec(F.read(), None, lcl)
    version = lcl['version']

if not version.is_release:
    print('Is pre-release')
    # https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions#setting-an-environment-variable
    #echo "{name}={value}" >> $GITHUB_ENV

    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a') as F:
            F.write('PRE=--pre\n')
    else:
        print('Would export PRE=--pre')
