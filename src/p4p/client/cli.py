
from __future__ import print_function

import logging
_log = logging.getLogger(__name__)

import sys
import time
import json
try:
    from itertools import izip
except ImportError:
    izip = zip

import logging
_log = logging.getLogger(__name__)

from .. import Value
from .. import nt
from .. import set_debug
from . import thread


def op_get(ctxt, args):
    requests = [args.request] * len(args.names)
    results = ctxt.get(args.names, requests, timeout=args.timeout, throw=False)
    ret = 0
    for name, val in izip(args.names, results):
        if isinstance(val, Exception):
            ret = 1
            print(name, 'Error:', val)
        else:
            print(name, val)
    sys.exit(ret)


def op_put(ctxt, args):
    requests = [args.request] * len(args.names)

    names, values = [], []
    for pair in args.names:
        N, sep, V = pair.partition('=')
        if sep is '':
            print("Missing expected '=' after", pair)
            sys.exit(1)
        elif V[:1] in '{[':
            V = json.loads(V)
        N = N.strip()
        _log.debug("put %s <- %s", N, V)
        names.append(N)
        values.append(V)

    results = ctxt.put(names, values, requests, timeout=args.timeout, throw=False)

    ret = 0
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
            if val is None:
                print(name, "Disconnect")
            elif isinstance(val, Exception):
                ret = 1
                print(name, 'Error:', val)
            else:
                print(name, val)
        subs.append(ctxt.monitor(name, show, args.request, notify_disconnect=True))

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        ret = 1
    [S.close() for S in subs]
    sys.exit(ret)


def op_rpc(ctxt, args):
    anames = []
    kws = {}
    for arg in args.args:
        K, sep, V = arg.partition('=')
        if not sep:
            print("arguments must be name=value not:", arg)
            sys.exit(2)
        elif V[:1] in '{[':
            V = json.loads(V)

        anames.append((K, 's'))
        kws[K] = V

    uri = nt.NTURI(anames).wrap(args.name, kws=kws)  # only keyword arguments

    ret = ctxt.rpc(args.name, uri, request=args.request, timeout=args.timeout, throw=False)
    if isinstance(ret, Exception):
        print('Error:', ret)
        sys.exit(1)
    else:
        print(ret.tolist())


def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('-r', '--request', default='')
    P.add_argument('-w', '--timeout', type=float, default=5.0)
    P.add_argument('-p', '--provider', default='pva')
    P.add_argument('-d', '--debug', action='store_const', const=logging.DEBUG, default=logging.INFO)
    P.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO)
    P.add_argument('--raw', action='store_false', default=None)
    P.set_defaults(func=lambda ctxt, args: P.print_help())

    SP = P.add_subparsers()

    PP = SP.add_parser('get')
    PP.add_argument('names', nargs='*')
    PP.set_defaults(func=op_get)

    PP = SP.add_parser('put')
    PP.add_argument('names', nargs='*', metavar='name=value', help='PV names and values')
    PP.set_defaults(func=op_put)

    PP = SP.add_parser('monitor')
    PP.add_argument('names', nargs='*')
    PP.set_defaults(func=op_monitor)

    PP = SP.add_parser('rpc')
    PP.add_argument('name')
    PP.add_argument('args', nargs='*')
    PP.set_defaults(func=op_rpc)

    return P.parse_args()


def main(args):
    with thread.Context(args.provider, unwrap=args.raw) as ctxt:
        args.func(ctxt, args)

if __name__ == '__main__':
    args = getargs()
    set_debug(args.debug)
    logging.basicConfig(level=args.verbose)
    main(args)
