from __future__ import print_function

import unittest, random, weakref, sys, gc, inspect, threading

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

from ..wrapper import Value, Type
from ..client.thread import Context, Disconnected, TimeoutError
from ..server import Server, StaticProvider
from ..server.thread import SharedPV, _defaultWorkQueue
from ..util import WorkQueue
from ..nt import NTScalar
from .utils import RefTestCase

class TestGPM(RefTestCase):
    maxDiff = 1000
    timeout = 1.0

    class Times2Handler(object):
        def put(self, pv, op):
            V = op.value()
            if V.raw.changed('value'):
                if V<0:
                    op.done(error="Must be non-negative")
                V = V *2
                pv.post(V)
            op.done()

    def setUp(self):
        #gc.set_debug(gc.DEBUG_LEAK)
        super(TestGPM, self).setUp()

        conf = {
            'EPICS_PVAS_INTF_ADDR_LIST':'127.0.0.1',
            'EPICS_PVA_ADDR_LIST':'127.0.0.1',
            'EPICS_PVA_AUTO_ADDR_LIST':'0',
            'EPICS_PVA_SERVER_PORT':'0',
            'EPICS_PVA_BROADCAST_PORT':'0',
        }

        self.pv = SharedPV(handler=self.Times2Handler(), nt=NTScalar('d'))
        self.pv2 = SharedPV(handler=self.Times2Handler(), nt=NTScalar('d'), initial=42.0)
        self.sprov = StaticProvider("serverend")
        self.sprov.add('foo', self.pv)
        self.sprov.add('bar', self.pv2)

        self.server = Server(providers=[self.sprov], conf=conf, useenv=False)

    def tearDown(self):
        self.server.stop()
        _defaultWorkQueue.stop()
        self.pv._handler._pv = None
        R = [weakref.ref(r) for r in (self.server, self.sprov, self.pv, self.pv._whandler, self.pv._handler)]
        r = None
        del self.server
        del self.sprov
        del self.pv
        del self.pv2
        gc.collect()
        R = [r() for r in R]
        self.assertListEqual(R, [None]*len(R))
        super(TestGPM, self).tearDown()

    def testGet(self):
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:
            # PV not yet opened
            self.assertRaises(TimeoutError, ctxt.get, 'foo', timeout=0.1)
            
            self.pv.open(1.0)

            # TODO: this really shouldn't fail, but does due to:
            # https://github.com/epics-base/pvAccessCPP/issues/103
            #  also proves that our Channel cache is working...
            self.assertRaises(RuntimeError, ctxt.get, 'foo', timeout=0.1)
            ctxt.disconnect('foo') # clear channel cache and force new channel to ensure we don't race to pick up the broken one

            V = ctxt.get('foo')
            self.assertEqual(V, 1.0)
            self.assertTrue(V.raw.changed('value'))

            self.assertEqual(ctxt.get(['foo', 'bar']), [1.0, 42.0])

        C = weakref.ref(ctxt)
        del ctxt
        gc.collect()
        self.assertIsNone(C())

    def testPutGet(self):
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:
            
            self.pv.open(1.0)

            V = ctxt.get('foo')
            self.assertEqual(V, 1.0)

            ctxt.put('foo', 5)

            V = ctxt.get('foo')
            self.assertEqual(V, 10.0)

            ctxt.put(['foo', 'bar'], [5, 6])

            self.assertEqual(ctxt.get(['foo', 'bar']), [5*2, 6*2])

        C = weakref.ref(ctxt)
        del ctxt
        gc.collect()
        self.assertIsNone(C())

    def testMonitor(self):
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:

            self.pv.open(1.0)

            Q = Queue(maxsize=4)
            sub = ctxt.monitor('foo', Q.put, notify_disconnect=True)

            V = Q.get(timeout=self.timeout)
            self.assertIsInstance(V, Disconnected)

            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 1.0)

            ctxt.put('foo', 4)

            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 8.0)

            self.pv.close()

            V = Q.get(timeout=self.timeout)
            self.assertIsInstance(V, Disconnected)

            self.pv.open(3.0)

            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 3.0)

        C = weakref.ref(ctxt)
        del ctxt
        del sub
        del Q
        gc.collect()
        self.assertIsNone(C())
