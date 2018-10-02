
import logging
import warnings
_log = logging.getLogger(__name__)

import unittest
import sys
import random
import weakref
import gc
from unittest.case import SkipTest

from ..nt import NTScalar, NTURI
from ..server import Server, StaticProvider
from .utils import RefTestCase

try:
    import cothread
except ImportError:
    raise SkipTest('No cothread')
else:
    from ..client.cothread import Context, Disconnected, RemoteError
    from ..server.cothread import SharedPV, _sync
    from ..server import cothread as srv_cothread

    class GPMHandler:
        def put(self, pv, op):
            _log.debug("putting %s <- %s", op.name(), op.value())
            cothread.Yield()  # because we can
            pv.post(op.value() * 2)
            op.done()

        def rpc(self, pv, op):
            V = op.value()
            if V.query.get('oops'):
                op.done(error='oops')
            else:
                op.done(NTScalar('i').wrap(42))

    class TestGPM(RefTestCase):
        timeout = 1.0

        def _sleep(self, delay):
            cothread.Sleep(delay)

        def setUp(self):
            super(TestGPM, self).setUp()

            self.pv = SharedPV(nt=NTScalar('i'), initial=0, handler=GPMHandler())
            self.pv2 = SharedPV(handler=GPMHandler(), nt=NTScalar('d'), initial=42.0)
            self.provider = StaticProvider("serverend")
            self.provider.add('foo', self.pv)
            self.provider.add('bar', self.pv2)

        def tearDown(self):
            del self.pv
            del self.pv2
            del self.provider
            _sync()
            gc.collect()
            self.assertSetEqual(set(srv_cothread._handlers), set()) # ensure no outstanding server handlers
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
                        self.assertIsInstance(Q.Wait(timeout=self.timeout), Disconnected)

                        self.assertEqual(0, Q.Wait(timeout=self.timeout))

                        C.put('foo', 2)

                        self.assertEqual(2 * 2, Q.Wait(timeout=self.timeout))

                        self.pv.close()

                        self.assertIsInstance(Q.Wait(timeout=self.timeout), Disconnected)

                        self.pv.open(3)

                        self.assertEqual(3, Q.Wait(timeout=self.timeout))

    class TestRPC(RefTestCase):
        timeout = 1.0

        def _sleep(self, delay):
            cothread.Sleep(delay)

        def setUp(self):
            super(TestRPC, self).setUp()

            self.pv = SharedPV(nt=NTScalar('i'), initial=0, handler=GPMHandler())
            self.provider = StaticProvider("serverend")
            self.provider.add('foo', self.pv)

        def tearDown(self):
            del self.pv
            del self.provider
            _sync()
            gc.collect()
            self.assertSetEqual(set(srv_cothread._handlers), set()) # ensure no outstanding server handlers
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

        def test_rpc_error(self):
            with Server(providers=[self.provider], isolate=True) as S:
                with Context('pva', conf=S.conf(), useenv=False) as C:

                    args = NTURI([
                        ('oops', '?'),
                    ])

                    with self.assertRaisesRegexp(RemoteError, 'oops'):
                        ret = C.rpc('foo', args.wrap('foo', kws={'oops':True}))

    class TestFirstLast(RefTestCase):
        maxDiff = 1000
        timeout = 1.0
        mode = 'Mask'

        class Handler:
            def __init__(self):
                self.evt = cothread.Event(auto_reset=False)
                self.conn = None
            def onFirstConnect(self, pv):
                _log.debug("onFirstConnect")
                self.conn = True
                self.evt.Signal()
            def onLastDisconnect(self, pv):
                _log.debug("onLastDisconnect")
                self.conn = False
                self.evt.Signal()

        def _sleep(self, delay):
            cothread.Sleep(delay)

        def setUp(self):
            # gc.set_debug(gc.DEBUG_LEAK)
            super(TestFirstLast, self).setUp()

            self.H = self.Handler()
            self.pv = SharedPV(handler=self.H,
                            nt=NTScalar('d'),
                            options={'mapperMode':self.mode})
            self.sprov = StaticProvider("serverend")
            self.sprov.add('foo', self.pv)

            self.server = Server(providers=[self.sprov], isolate=True)

        def tearDown(self):
            self.server.stop()
            del self.server
            del self.sprov
            del self.pv
            del self.H
            _sync()
            gc.collect()
            self.assertSetEqual(set(srv_cothread._handlers), set()) # ensure no outstanding server handlers
            super(TestFirstLast, self).tearDown()

        def testClientDisconn(self):
            self.pv.open(1.0)

            with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
                Q = cothread.EventQueue()
                with ctxt.monitor('foo', Q.Signal, notify_disconnect=True):

                    Q.Wait(timeout=self.timeout) # initial update

                    _log.debug('TEST')
                    self.H.evt.Wait(self.timeout) # onFirstConnect()
                    self.H.evt.Reset()
                    self.assertTrue(self.H.conn)

            self.H.evt.Wait(self.timeout) # onLastDisconnect()
            _log.debug('SHUTDOWN')
            self.assertFalse(self.H.conn)

        def testServerShutdown(self):
            self.pv.open(1.0)

            with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
                Q = cothread.EventQueue()
                with ctxt.monitor('foo', Q.Signal, notify_disconnect=True):

                    Q.Wait(timeout=self.timeout)

                    _log.debug('TEST')
                    self.H.evt.Wait(self.timeout) # initial update
                    self.H.evt.Reset()
                    self.assertIs(self.H.conn, True)

                    self.server.stop()

                    self.H.evt.Wait(self.timeout) # wait for
                    _log.debug('SHUTDOWN')
                    self.assertIs(self.H.conn, False)

        def testPVClose(self):
            self.pv.open(1.0)

            with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
                Q = cothread.EventQueue()
                with ctxt.monitor('foo', Q.Signal, notify_disconnect=True):

                    Q.Wait(timeout=self.timeout) # initial update

                    _log.debug('TEST')
                    self.H.evt.Wait(self.timeout)
                    self.H.evt.Reset()
                    self.assertTrue(self.H.conn)

                    _log.debug('CLOSING')
                    self.sprov.remove('foo') # prevent new connections while destroying
                    self.pv.close(destroy=True, sync=True, timeout=self.timeout)

                    _log.debug('CLOSED')
                    self.assertFalse(self.H.conn)
