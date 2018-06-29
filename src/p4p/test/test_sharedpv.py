from __future__ import print_function

import unittest, random, weakref, sys, gc, inspect, threading

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

from ..wrapper import Value, Type
from ..client.thread import Context, TimeoutError
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
            if V.changed('value'):
                if V.value<0:
                    op.done(error="Must be non-negative")
                V.value = V.value *2
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

        self.pv = SharedPV(handler=self.Times2Handler())
        self.sprov = StaticProvider("serverend")
        self.sprov.add('foo', self.pv)

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
        gc.collect()
        from sys import getrefcount
        frame = inspect.currentframe()
        for r in R:
            obj = r()
            if obj is None:
                continue
            print("XXX", getrefcount(obj)-1, obj)
            for A in gc.get_referrers(obj):
                if A is frame:
                    print(">>>X")
                    continue
                print(">>>", type(A), A)
                for B in gc.get_referrers(R):
                    if B is frame:
                        continue
                    print(">>>>>", type(B), B)
            del A
            del B
            gc.collect()

        R = [r() for r in R]
        self.assertListEqual(R, [None]*len(R))
        super(TestGPM, self).tearDown()

    def testGet(self):
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:
            # PV not yet opened
            self.assertRaises(TimeoutError, ctxt.get, 'foo', timeout=0.1)
            
            type = NTScalar('d')
            self.pv.open(type.wrap(1.0))

            # TODO: this really shouldn't fail, but does due to:
            # https://github.com/epics-base/pvAccessCPP/issues/103
            #  also proves that our Channel cache is working...
            self.assertRaises(RuntimeError, ctxt.get, 'foo', timeout=0.1)

            V = ctxt.get('foo')
            self.assertEqual(V, 1.0)
            self.assertTrue(V.raw.changed('value'))

        C = weakref.ref(ctxt)
        del ctxt
        gc.collect()
        self.assertIsNone(C())

    def testPutGet(self):
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:
            
            type = NTScalar('d')
            self.pv.open(type.wrap(1.0))

            V = ctxt.get('foo')
            self.assertEqual(V, 1.0)

            ctxt.put('foo', 5)

            V = ctxt.get('foo')
            self.assertEqual(V, 10.0)

        C = weakref.ref(ctxt)
        del ctxt
        gc.collect()
        self.assertIsNone(C())

    def testMonitor(self):
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:

            type = NTScalar('d')
            self.pv.open(type.wrap(1.0))

            Q = Queue(maxsize=4)
            sub = ctxt.monitor('foo', Q.put)

            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 1.0)

            ctxt.put('foo', 4)

            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 8.0)

        C = weakref.ref(ctxt)
        del ctxt
        del sub
        del Q
        gc.collect()
        self.assertIsNone(C())
