#!/usr/bin/env python
"""Example demonstrating RPC server

Single threaded request handling.
Additional requests are queued.

  $ ./rpc_server.py example:

To handle 2 requests concurrently.

  $ ./rpc_server.py example: --worker 2

test with

  $ eget -s example:add -a lhs=1 -a rhs=1
  2

  $ eget -s example:echo -a value=2
  2

or rpc_client.py

"""

import time, logging

from p4p.rpc import rpc, quickRPCServer
from p4p.nt import NTScalar

class MyExample(object):
    @rpc(NTScalar("d"))
    def add(self, lhs, rhs):
        return float(lhs) + float(rhs)

    @rpc(NTScalar("s"))
    def echo(self, value, delay=1):
        print("Start echo", value,"wait",delay)
        time.sleep(float(delay))
        print("End echo", value,"wait",delay)
        return value

example = MyExample()

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('--workers', type=int, default=1)
    P.add_argument('-d','--debug', action='store_true', default=False)
    P.add_argument('prefix')
    return P.parse_args()

args = getargs()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

try:
    # "Example" is an arbitrary name, which must be unique
    # within this process (not globally).
    quickRPCServer(provider="Example", 
                   prefix=args.prefix,
                   workers=args.workers,
                   target=example)
except KeyboardInterrupt:
    pass
