
from __future__ import print_function

import sys
from itertools import izip

import logging
_log = logging.getLogger(__name__)

from . import thread

def op_get(ctxt, args):
    results = ctxt.get(args.names, throw=False)
    ret= 0
    for name, val in izip(args.names, results):
        if isinstance(val, Exception):
            ret = 1
            print(name, val)
        else:
            print(name, val.tolist())
    sys.exit(ret)

def op_put(ctxt, args):
    names, values = [], []
    for pair in args.names:
        N, sep, V = pair.partition('=')
        if sep is None:
            print("Missing expected '=' after", pair)
            sys.exit(1)
        elif V is None:
            V = ''

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('-r', '--request', default='')
    P.add_argument('-w', '--timeout', type=float, default=5.0)
    P.add_argument('-p', '--provider', default='pva')
    P.add_argument('-d','--debug', action='store_true')

    SP = P.add_subparsers()

    PP = SP.add_parser('get')
    PP.add_argument('names', nargs='*')
    PP.set_defaults(func=op_get)

    PP = SP.add_parser('put')
    PP.add_argument('names', nargs='*')
    PP.set_defaults(func=op_put)

    return P.parse_args()

def main(args):
    with thread.Context(args.provider) as ctxt:
        args.func(ctxt, args)

if __name__=='__main__':
    args = getargs()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    main(args)
