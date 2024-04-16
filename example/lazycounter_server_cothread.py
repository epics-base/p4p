#!/usr/bin/env python
"""A counter which only increments while at least one client is connected.

As a demonstration of the ability to perform type changes,
switch between integer to float each time the counter is stopped.

All the real work, including type change, is performed from a Timer
to show that this can be done asynchronously.

   $ pvget -m foo
"""

import time, logging
_log = logging.getLogger(__name__)

import cothread

from p4p.nt import NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.cothread import SharedPV

logging.basicConfig(level=logging.DEBUG)

types = {
    False:NTScalar('I'),
    True:NTScalar('d'),
}

class LazyCounter(object):
    def __init__(self):
        self.timer = None
        self.count = 0
        self.pv = None
        self.select = False
        self.active = False

    def onFirstConnect(self, pv):
        _log.info("First client connects")
        if self.timer is None:
            # start timer if necessary
            self.timer = cothread.Timer(1.0, self._tick, retrigger=True)
        self.pv = pv
        self.active = True

    def _tick(self):
        if not self.active:
            _log.info("Close")
            # no clients connected
            if self.pv.isOpen():
                self.pv.close()
                self.select = not self.select # toggle type for next clients

            # cancel timer until a new first client arrives
            self.timer.cancel()
            self.pv = self.timer = None

        else:
            NT = types[self.select]

            if not self.pv.isOpen():
                _log.info("Open %s", self.count)
                self.pv.open(NT.wrap(self.count))

            else:
                _log.info("Tick %s", self.count)
                self.pv.post(NT.wrap(self.count))
            self.count += 1

    def onLastDisconnect(self, pv):
        _log.info("Last client disconnects")
        # mark in-active, but don't immediately close()
        self.active = False

    def put(self, pv, op):
        # force counter value
        self.count = op.value().value
        op.done()

pv = SharedPV(handler=LazyCounter())

with Server(providers=[{'foo': pv}]):
    print('Running')
    try:
        cothread.WaitForQuit()
    except KeyboardInterrupt:
        pass

print('Done')
