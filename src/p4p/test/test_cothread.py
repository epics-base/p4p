
import logging
import warnings
_log = logging.getLogger(__name__)

import unittest
import sys
import random
import weakref
import gc
import threading
from unittest.case import SkipTest

from ..nt import NTScalar, NTURI
from ..server import Server, StaticProvider
from .utils import RefTestCase

try:
    import cothread
except ImportError:
    raise SkipTest('No cothread')
else:
    from ..client.cothread import Context, Disconnected
    from ..server.cothread import SharedPV

    class Handler:
        def __init__(self):
            self.lastrpc = None

        def put(self, pv, op):
            _log.debug("putting %s <- %s", op.name(), op.value())
            cothread.Yield()  # because we can
            pv.post(op.value() * 2)
            op.done()

        def rpc(self, pv, op):
            self.lastrpc = op.value()
            op.done(NTScalar('i').wrap(42))

    class TestGPM(RefTestCase):
        def _sleep(self, delay):
            cothread.Sleep(delay)

        def setUp(self):
            super(TestGPM, self).setUp()

            self.pv = SharedPV(nt=NTScalar('i'), initial=0, handler=Handler())
            self.wpv = weakref.ref(self.pv)
            self.pv2 = SharedPV(handler=Handler(), nt=NTScalar('d'), initial=42.0)
            self.provider = StaticProvider("serverend")
            self.provider.add('foo', self.pv)
            self.provider.add('bar', self.pv2)

        def tearDown(self):
            self.pv.close(destroy=True)
            self.pv2.close(destroy=True)
            del self.pv
            del self.pv2
            del self.provider
            gc.collect()
            pv = self.wpv()
            if pv is not None:
                _log.error("Live PV %s %s", sys.getrefcount(pv), gc.get_referrers(pv))
            del pv
            super(TestGPM, self).tearDown()

        def test_getput(self):
            with Server(providers=[self.provider], isolate=True) as S:
                with Context('pva', conf=S.conf(), useenv=False) as C:
                    self.assertEqual(0, C.get('foo'))

                    C.put('foo', 5)

                    self.assertEqual(5 * 2, C.get('foo'))

                    self.assertEqual([5 * 2, 42.0], C.get(['foo', 'bar']))

                    C.put(['foo', 'bar'], [6, 7])

                    self.assertEqual([6 * 2, 7 * 2], C.get(['foo', 'bar']))

        def test_monitor(self):
            with Server(providers=[self.provider], isolate=True) as S:
                with Context('pva', conf=S.conf(), useenv=False) as C:

                    Q = cothread.EventQueue()

                    with C.monitor('foo', Q.Signal, notify_disconnect=True) as sub:
                        self.assertIsInstance(Q.Wait(), Disconnected)

                        self.assertEqual(0, Q.Wait())

                        C.put('foo', 2)

                        self.assertEqual(2 * 2, Q.Wait())

                        self.pv.close()

                        self.assertIsInstance(Q.Wait(), Disconnected)

                        self.pv.open(3)

                        self.assertEqual(3, Q.Wait())

    class TestRPC(RefTestCase):
        def _sleep(self, delay):
            cothread.Sleep(delay)

        def setUp(self):
            super(TestRPC, self).setUp()

            self.pv = SharedPV(nt=NTScalar('i'), initial=0, handler=Handler())
            self.provider = StaticProvider("serverend")
            self.provider.add('foo', self.pv)

        def tearDown(self):
            self.pv.close(destroy=True)
            del self.pv
            del self.provider
            super(TestRPC, self).tearDown()

        def test_rpc(self):
            with Server(providers=[self.provider], isolate=True) as S:
                with Context('pva', conf=S.conf(), useenv=False) as C:

                    args = NTURI([
                        ('lhs', 'd'),
                        ('rhs', 'd'),
                    ])

                    ret = C.rpc('foo', args.wrap('foo', kws={'lhs':1, 'rhs':2}))
                    _log.debug("RET %s", ret)
                    self.assertEqual(ret, 42)
