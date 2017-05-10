
from __future__ import print_function

import sys, time
try:
    from itertools import izip
except ImportError:
    izip = zip

import logging
_log = logging.getLogger(__name__)

from . import thread

def op_get(ctxt, args):
    requests = [args.request]*len(args.names)
    results = ctxt.get(args.names, requests, throw=False)
    ret= 0
    for name, val in izip(args.names, results):
        if isinstance(val, Exception):
            ret = 1
            print(name, 'Error:', val)
        else:
            print(name, val.tolist())
    sys.exit(ret)

def op_put(ctxt, args):
    requests = [args.request]*len(args.names)

    names, values = [], []
    for pair in args.names:
        N, sep, V = pair.partition('=')
        if sep is None:
            print("Missing expected '=' after", pair)
            sys.exit(1)
        elif V is None:
            V = ''
        N = N.strip()
        names.append(N)
        values.append(V)

    results = ctxt.put(names, values, requests, throw=False)

    ret= 0
    for name, val in izip(args.names, results):
        if isinstance(val, Exception):
            ret = 1
            print(name, 'Error:', val)
        elif val is None:
            print(name, 'ok')
        else:
            print(name, val.tolist())
    sys.exit(ret)

def op_monitor(ctxt, args):

    subs = []
    ret = 0
    for name in args.names:
        def show(val, name=name):
            if isinstance(val, Exception):
                ret = 1
                print(name, 'Error:', val)
            else:
                print(name, val.tolist())
        subs.append(ctxt.monitor(name, show, args.request))

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        ret = 1
    [S.close() for S in subs]
    sys.exit(ret)

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

    PP = SP.add_parser('monitor')
    PP.add_argument('names', nargs='*')
    PP.set_defaults(func=op_monitor)

    return P.parse_args()

def main(args):
    with thread.Context(args.provider) as ctxt:
        args.func(ctxt, args)

if __name__=='__main__':
    args = getargs()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    main(args)
