#!/usr/bin/env python

"""Bootstrap pvagw executable
"""

import sys
import os
import warnings

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('output')
    P.add_argument('-P', '--pythonpath')
    return P

def main(args):
    if 'VIRTUAL_ENV' in os.environ:
        warnings.warn('virtualenv %s will be required to run %s'%(os.environ['VIRTUAL_ENV'], args.output))

    with open(args.output, 'w') as F:
        F.write("""#!{exe}
import sys
import os

mypath = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(mypath, '{pythonpath}'))

from p4p.gw import main
main()

""".format(exe=sys.executable,
           pythonpath=args.pythonpath))

if __name__=='__main__':
    args = getargs().parse_args()
    main(args)
