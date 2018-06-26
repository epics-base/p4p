#!/usr/bin/env python
"""Motor simulation

pvput -w 10 foo 4
"""

from __future__ import print_function

import time, logging

import cothread

from p4p.nt import NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.cothread import SharedPV

logging.basicConfig(level=logging.DEBUG)

class MoveHandler(object):
    def __init__(self):
        self.pos = 0
        self.busy = False

    def put(self, pv, op):
        if self.busy:
            op.done(error="Move in progress")
            return
        self.busy = True
        try:
            initial = self.pos
            final = op.value().value
            delta = abs(final-initial)
            op.info("Moving %s -> %s"%(initial, final))

            while delta>=1.0:
                op.info("Moving %s"%delta)
                delta -= 1.0
                cothread.Sleep(1.0) # move at 1 step per second

            self.pos = final
            op.done()
        finally:
            self.busy = False

pv = SharedPV(initial=NTScalar('d').wrap(0),
              handler=MoveHandler())

provider = StaticProvider('move') # 'move' is an arbitrary name
provider.add("foo", pv)

with Server(providers=[provider]):
    print('Running')
    try:
        cothread.WaitForQuit()
    except KeyboardInterrupt:
        pass

print('Done')
