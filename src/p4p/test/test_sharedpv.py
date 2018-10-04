from __future__ import print_function

import logging
import unittest
import random
import weakref
import sys
import gc
import inspect
import threading

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

from ..wrapper import Value, Type
from ..client.thread import Context, Disconnected, TimeoutError, RemoteError
from ..server import Server, StaticProvider
from ..server.thread import SharedPV, _defaultWorkQueue
from ..util import WorkQueue
from ..nt import NTScalar, NTURI
from .utils import RefTestCase

_log = logging.getLogger(__name__)

class TestGPM(RefTestCase):
    maxDiff = 1000
    timeout = 1.0

    class Times2Handler(object):

        def put(self, pv, op):
            V = op.value()
            if V.raw.changed('value'):
                if V < 0:
                    op.done(error="Must be non-negative")
                V = V * 2
                pv.post(V)
            op.done()

    def setUp(self):
        # gc.set_debug(gc.DEBUG_LEAK)
        super(TestGPM, self).setUp()

        self.pv = SharedPV(handler=self.Times2Handler(), nt=NTScalar('d'))
        self.pv2 = SharedPV(handler=self.Times2Handler(), nt=NTScalar('d'), initial=42.0)
        self.sprov = StaticProvider("serverend")
        self.sprov.add('foo', self.pv)
        self.sprov.add('bar', self.pv2)

        self.server = Server(providers=[self.sprov], isolate=True)

    def tearDown(self):
        self.server.stop()
        _defaultWorkQueue.sync()
        #self.pv._handler._pv = None
        R = [weakref.ref(r) for r in (self.server, self.sprov, self.pv, self.pv._whandler, self.pv._handler)]
        r = None
        del self.server
        del self.sprov
        del self.pv
        del self.pv2
        gc.collect()
        R = [r() for r in R]
        self.assertListEqual(R, [None] * len(R))
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
            ctxt.disconnect('foo')  # clear channel cache and force new channel to ensure we don't race to pick up the broken one

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

            self.assertEqual(ctxt.get(['foo', 'bar']), [5 * 2, 6 * 2])

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

class TestRPC(RefTestCase):
    maxDiff = 1000
    timeout = 1.0
    openclose = False

    class Handler:
        def __init__(self, openclose):
            self.openclose = openclose
        def onFirstConnect(self, pv):
            _log.debug("onFirstConnect")
            if self.openclose:
                pv.open(0)

        def onLastDisconnect(self, pv):
            _log.debug("onLastDisconnect")
            if self.openclose:
                pv.close()

        def rpc(self, pv, op):
            V = op.value()
            if V.get('query.oops'):
                op.done(error='oops')
            elif V.get('query.null'):
                op.done()
            else:
                op.done(NTScalar('i').wrap(42))

    def setUp(self):
        super(TestRPC, self).setUp()

        self.pv = SharedPV(nt=NTScalar('i'), handler=self.Handler(self.openclose))
        self.provider = StaticProvider("serverend")
        self.provider.add('foo', self.pv)

    def tearDown(self):
        self.pv.close(sync=True, timeout=self.timeout)
        self.traceme(self.pv)
        self.traceme(self.provider)
        del self.pv
        del self.provider
        _defaultWorkQueue.sync()
            
        super(TestRPC, self).tearDown()

    def test_rpc(self):
        with Server(providers=[self.provider], isolate=True) as S:
            with Context('pva', conf=S.conf(), useenv=False) as C:

                args = NTURI([
                    ('lhs', 'd'),
                    ('rhs', 'd'),
                ])

                # self.pv not open()'d
                ret = C.rpc('foo', args.wrap('foo', kws={'lhs':1, 'rhs':2}))
                _log.debug("RET %s", ret)
                self.assertEqual(ret, 42)

                ret = C.rpc('foo', None)
                _log.debug("RET %s", ret)
                self.assertEqual(ret, 42)

    def test_rpc_null(self):
        with Server(providers=[self.provider], isolate=True) as S:
            with Context('pva', conf=S.conf(), useenv=False) as C:

                args = NTURI([
                    ('null', '?'),
                ])

                # self.pv not open()'d
                ret = C.rpc('foo', args.wrap('foo', kws={'null':True}))
                _log.debug("RET %s", ret)
                #self.assertIsNone(ret)
                self.assertIsInstance(ret, Value)
                self.assertListEqual(ret.keys(), [])

    def test_rpc_error(self):
        with Server(providers=[self.provider], isolate=True) as S:
            with Context('pva', conf=S.conf(), useenv=False) as C:

                args = NTURI([
                    ('oops', '?'),
                ])

                with self.assertRaisesRegexp(RemoteError, 'oops'):
                    ret = C.rpc('foo', args.wrap('foo', kws={'oops':True}))

class TestRPC2(TestRPC):
    openclose = True

class TestPVRequestMask(RefTestCase):
    maxDiff = 1000
    timeout = 1.0
    mode = 'Mask'

    class Handler(object):
        def put(self, pv, op):
            val = op.value()
            _log.debug("Putting %s %s", val.raw.changedSet(), str(val.raw))
            pv.post(op.value())
            op.done()

    def setUp(self):
        # gc.set_debug(gc.DEBUG_LEAK)
        super(TestPVRequestMask, self).setUp()

        self.pv = SharedPV(handler=self.Handler(),
                           nt=NTScalar('d'),
                           initial=1.0,
                           options={'mapperMode':self.mode})
        self.sprov = StaticProvider("serverend")
        self.sprov.add('foo', self.pv)

        self.server = Server(providers=[self.sprov], isolate=True)

    def tearDown(self):
        self.server.stop()
        _defaultWorkQueue.sync()
        self.pv._handler._pv = None
        R = [weakref.ref(r) for r in (self.server, self.sprov, self.pv, self.pv._whandler, self.pv._handler)]
        r = None
        del self.server
        del self.sprov
        del self.pv
        gc.collect()
        R = [r() for r in R]
        self.assertListEqual(R, [None] * len(R))
        super(TestPVRequestMask, self).tearDown()

    def testGetPut(self):
        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
            V = ctxt.get('foo', request='value')

            self.assertEqual(V.value, 1.0)

            self.assertTrue(V.changed('value'))

            if self.mode=='Mask':
                self.assertSetEqual(V.changedSet(), {'value'})
            else:
                self.assertListEqual(V.keys(), ['value'])

            ctxt.put('foo', {'value':2.0}, request='value')

            V = ctxt.get('foo', request='value')

            self.assertEqual(V.value, 2.0)

    def testMonitor(self):
        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:

            Q = Queue(maxsize=4)
            sub = ctxt.monitor('foo', Q.put, request='value')

            V = Q.get(timeout=self.timeout)
            self.assertEqual(V.value, 1.0)

            if self.mode=='Mask':
                self.assertSetEqual(V.changedSet(), {'value'})
            else:
                self.assertListEqual(V.keys(), ['value'])

            ctxt.put('foo', {'alarm.severity':1}) # should be dropped

            ctxt.put('foo', {'value':3.0}, request='value')

            V = Q.get(timeout=self.timeout)
            self.assertEqual(V.value, 3.0)

            if self.mode=='Mask':
                self.assertSetEqual(V.changedSet(), {'value'})
            else:
                self.assertListEqual(V.keys(), ['value'])


class TestPVRequestSlice(TestPVRequestMask):
    mode = 'Slice'

class TestFirstLast(RefTestCase):
    maxDiff = 1000
    timeout = 1.0
    mode = 'Mask'

    class Handler:
        def __init__(self):
            self.evt = threading.Event()
            self.conn = None
        def onFirstConnect(self, pv):
            _log.debug("onFirstConnect")
            self.conn = True
            self.evt.set()
        def onLastDisconnect(self, pv):
            _log.debug("onLastDisconnect")
            self.conn = False
            self.evt.set()

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
        _defaultWorkQueue.sync()
        del self.server
        del self.sprov
        del self.pv
        del self.H
        super(TestFirstLast, self).tearDown()

    def testClientDisconn(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
            Q = Queue(maxsize=4)
            with ctxt.monitor('foo', Q.put, notify_disconnect=True):

                Q.get(timeout=self.timeout) # initial update

                _log.debug('TEST')
                self.H.evt.wait(self.timeout) # onFirstConnect()
                self.H.evt.clear()
                self.assertTrue(self.H.conn)

        self.H.evt.wait(self.timeout) # onLastDisconnect()
        _log.debug('SHUTDOWN')
        self.assertFalse(self.H.conn)

    def testServerShutdown(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
            Q = Queue(maxsize=4)
            with ctxt.monitor('foo', Q.put, notify_disconnect=True):

                Q.get(timeout=self.timeout) # initial update

                _log.debug('TEST')
                self.H.evt.wait(self.timeout) # onFirstConnect()
                self.H.evt.clear()
                self.assertIs(self.H.conn, True)

                self.server.stop()

                self.H.evt.wait(self.timeout) # onLastDisconnect()
                _log.debug('SHUTDOWN')
                self.assertIs(self.H.conn, False)

    def testPVClose(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
            Q = Queue(maxsize=4)
            with ctxt.monitor('foo', Q.put, notify_disconnect=True):

                Q.get(timeout=self.timeout) # initial update

                _log.debug('TEST')
                self.H.evt.wait(self.timeout) # onFirstConnect()
                self.H.evt.clear()
                self.assertTrue(self.H.conn)

                self.pv.close(destroy=True, sync=True, timeout=self.timeout) # onLastDisconnect()

                _log.debug('CLOSE')
                self.assertFalse(self.H.conn)
