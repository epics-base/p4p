#!/usr/bin/env python


import time, sys, logging

from p4p.rpc import rpccall, rpcproxy
from p4p.client.thread import Context

@rpcproxy
class ExampleProxy(object):
    @rpccall("%sadd")
    def add(lhs='d', rhs='d'):
        pass
    @rpccall('%secho')
    def echo(value='s', delay='d'):
        pass

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('-d','--debug', action='store_true', default=False)
    P.add_argument('prefix')
    P.add_argument('method')
    P.add_argument('args', nargs='*')
    return P, P.parse_args()

P, args = getargs()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

ctxt = Context('pva')

proxy = ExampleProxy(context=ctxt, format=args.prefix)

if args.method=='add':
    print(proxy.add(*args.args[:2]))
elif args.method=='echo':
    print(proxy.echo(*args.args[:2]))
else:
    print("No method", P.method)
    sys.exit(1)
