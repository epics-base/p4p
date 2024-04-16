#!/usr/bin/env python
"""Demo server providing a set of PVs which do nothing except store a value.

These mailbox PVs can be dynamically created and destroyed via RPC calls.

   $ python dynamicbox_server.py foo:


   $ eget -s foo:add -a name=bar -a type=float
   $ pvinfo bar
   $ eget -s foo:del -a name=bar

   The list of mailbox PVs can be tracked with:

   $ pvget -m foo:list
"""

from __future__ import print_function

import sys
import time, logging

_log = logging.getLogger(__name__)

from threading import Lock

from p4p.nt import NTScalar, NTEnum
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV

logging.basicConfig(level=logging.DEBUG)

prefix = sys.argv[1]

list_type = NTScalar('as')

types = {
    'int':NTScalar('i').wrap(0),
    'float':NTScalar('d').wrap(0.0),
    'str':NTScalar('s').wrap(''),
    'enum':NTEnum().wrap(0),
}

pvs_lock = Lock()
pvs = {}

provider = StaticProvider('dynamicbox')


class MailboxHandler(object):
    def put(self, pv, op):
        # allow client to modify all fields.  eg. including .timeStamp
        pv.post(op.value())
        op.done()

addpv = SharedPV(initial=NTScalar('s').wrap('Only RPC'))
delpv = SharedPV(initial=NTScalar('s').wrap('Only RPC'))
listpv = SharedPV(nt=list_type, initial=[])

provider.add(prefix + "add", addpv)
provider.add(prefix + "del", delpv)
provider.add(prefix + "list", listpv)
_log.info("add with %s, remove with %s, list with %s", prefix + "add", prefix + "del", prefix + "list")

@addpv.rpc
def adder(pv, op):
    name = op.value().query.name
    type = op.value().query.get('type', 'int')

    if type not in types:
        op.done(error='unknown type %s.  Known types are %s'%(type, ', '.join(types)))
        return

    with pvs_lock:

        if name in pvs:
            op.done(error='PV already exists')
            return

        pv = SharedPV(initial=types[type], handler=MailboxHandler())
        provider.add(name, pv)
        pvs[name] = pv
        names = list(pvs) # makes a copy to ensure consistency outside lock

    _log.info("Added mailbox %s", name)
    listpv.post(names)
    op.done()


@delpv.rpc
def remover(pv, op):
    name = op.value().query.name

    with pvs_lock:
        if name not in pvs:
            op.done(error="PV doesn't exists")
            return
        pv = pvs.pop(name)
        provider.remove(name)
        names = list(pvs) # makes a copy to ensure consistency outside lock

    _log.info("Removed mailbox %s", name)
    listpv.post(names)

    op.done()

Server.forever(providers=[provider])

print('Done')
