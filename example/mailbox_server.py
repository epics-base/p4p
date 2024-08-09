#!/usr/bin/env python
"""Demo server providing a set of PVs which do nothing except store a value.

For fun, allow type change via an RPC.


   $ python example/mailbox_server.py foo

In another shell

   $ pvinfo foo
   $ eget -s foo -a help
   $ eget -s foo -a newtype=str
   $ pvinfo foo
"""

import time, logging

from p4p.nt import NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV

help_type = NTScalar('s')
types = {
    'int':NTScalar('i').wrap(0),
    'float':NTScalar('d').wrap(0.0),
    'str':NTScalar('s').wrap(''),
}

class MailboxHandler(object):
    type = None
    def rpc(self, pv, op):
        V = op.value()
        print("RPC", V, V.query.get('help'), V.query.get('newtype'))
        if V.query.get('help') is not None:
            op.done(help_type.wrap('Try newtype=int (or float or str)'))
            return

        newtype = types[V.query.newtype]

        op.done(help_type.wrap('Success'))

        pv.close() # disconnect client
        pv.open(newtype)

    def put(self, pv, op):
        val = op.value()
        logging.info("Assign %s = %s", op.name(), val)
        # Notify any subscribers of the new value.
        # Also set timeStamp with current system time.
        pv.post(val, timestamp=time.time())
        # Notify the client making this PUT operation that it has now completeted
        op.done()

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('name', nargs='+')
    P.add_argument('-v','--verbose', action='store_const', default=logging.INFO, const=logging.DEBUG)
    return P.parse_args()

def main(args):
    provider = StaticProvider('mailbox') # 'mailbox' is an arbitrary name

    pvs = [] # we must keep a reference in order to keep the Handler from being collected
    for name in args.name:
        pv = SharedPV(initial=types['int'], handler=MailboxHandler())

        provider.add(name, pv)
        pvs.append(pv)

    Server.forever(providers=[provider])

    print('Done')
if __name__=='__main__':
    args = getargs()
    logging.basicConfig(level=args.verbose)
    main(args)
