from __future__ import print_function

import logging
import unittest
import random
from unittest.mock import patch
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
from ..server.raw import Handler
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
        _log.debug('Server Conf: %s', self.server.conf())

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

    def testCurrent(self):
            self.pv.open(1.0)
            self.assertEqual(self.pv.current(), 1.0)

    def testGet(self):
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:
            _log.debug('Client conf: %s', ctxt.conf())
            # PV not yet opened
            self.assertRaises(TimeoutError, ctxt.get, 'foo', timeout=0.1)

            self.pv.open(1.0)

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
            ctxt.hurryUp()

            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 3.0)

        C = weakref.ref(ctxt)
        del ctxt
        del sub
        del Q
        gc.collect()
        self.assertIsNone(C())

class TestGPMNewHandler(RefTestCase):
    maxDiff = 1000
    timeout = 1.0

    class Times3Handler(Handler):
        # Note that the prototypes of open() and post() return None and here
        # we have them returning bool. A more rigorous solution might use 
        # Exceptions instead, which would also allow an error message to be 
        # passed to the client via the op.done()
        def open(self, value):
            if value.changed('value'):
                if value["value"] < 0:
                    value.unmark()
                    return False
                value["value"] = value["value"] * 3

            return True

        def post(self, pv, value, **kws):
            if not kws.pop("handler_post_enable", True):
                return
            
            return self.open(value)
        
        def put(self, pv, op):
            if not self.post(pv, op.value().raw):
                op.done(error="Must be non-negative")
            pv.post(op.value(),handler_post_enable=False)
            op.done()

    def setUp(self):
        # gc.set_debug(gc.DEBUG_LEAK)
        super(TestGPMNewHandler, self).setUp()

        self.pv = SharedPV(handler=self.Times3Handler(), nt=NTScalar('d'))
        self.pv2 = SharedPV(handler=self.Times3Handler(), nt=NTScalar('d'), initial=42.0)
        self.sprov = StaticProvider("serverend")
        self.sprov.add('foo', self.pv)
        self.sprov.add('bar', self.pv2)

        self.server = Server(providers=[self.sprov], isolate=True)
        _log.debug('Server Conf: %s', self.server.conf())

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
        super(TestGPMNewHandler, self).tearDown()

    def testCurrent(self):
        self.pv.open(1.0)
        self.assertEqual(self.pv.current(), 3.0)

    def testGet(self):
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:
            _log.debug('Client conf: %s', ctxt.conf())
            # PV not yet opened
            self.assertRaises(TimeoutError, ctxt.get, 'foo', timeout=0.1)

            self.pv.open(1.0)

            V = ctxt.get('foo')
            self.assertEqual(V, 3.0)
            self.assertTrue(V.raw.changed('value'))

            self.assertEqual(ctxt.get(['foo', 'bar']), [1.0 * 3, 42.0 * 3])

        C = weakref.ref(ctxt)
        del ctxt
        gc.collect()
        self.assertIsNone(C())

    def testPutGet(self):
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:

            self.pv.open(1.0)

            V = ctxt.get('foo')
            self.assertEqual(V, 3.0)

            ctxt.put('foo', 5)

            V = ctxt.get('foo')
            self.assertEqual(V, 15.0)

            ctxt.put(['foo', 'bar'], [5, 6])

            self.assertEqual(ctxt.get(['foo', 'bar']), [5 * 3, 6 * 3])

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
            self.assertEqual(V, 3.0)

            ctxt.put('foo', 4)

            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 12.0)

            self.pv.close()

            V = Q.get(timeout=self.timeout)
            self.assertIsInstance(V, Disconnected)

            self.pv.open(3.0)
            ctxt.hurryUp()

            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 9.0)

        C = weakref.ref(ctxt)
        del ctxt
        del sub
        del Q
        gc.collect()
        self.assertIsNone(C())


class TestNewHandler(RefTestCase):
    maxDiff = 1000
    timeout = 1.0

    class ValueChangeHandler(Handler):
        """Check whether calls work as expected"""
        put_use_handler_post = True

        def open(self, value, **kwargs):
            if not kwargs.pop("handler_open_enable", True):
                return
            
            value["value"] = 1.1

        def onFirstConnect(self, pv):
            # Not intended to allow changes to pv?
            pass
        
        def onLastDisconnect(self, pv):
            # Not intended to allow changes to pv?
            pass

        def post(self, pv, value, **kwargs):
            if not kwargs.pop("handler_post_enable", True):
                return
            
            value["value"] = 2.2

        def put(self, pv, op):
            op.value().raw["value"] = 3.3
            pv.post(op.value(), handler_post_enable=self.put_use_handler_post)
            op.done()

        def close(self, pv):
            # Not intended to allow changes to pv?
            pass

    def setUp(self):
        # gc.set_debug(gc.DEBUG_LEAK)
        super(TestNewHandler, self).setUp()

        self.pv = SharedPV(nt=NTScalar('d'), handler=self.ValueChangeHandler())
        self.pv2 = SharedPV(nt=NTScalar('d'), initial=42.0)  # No handler
        self.sprov = StaticProvider("serverend")
        self.sprov.add('foo', self.pv)
        self.sprov.add('bar', self.pv2)

        self.server = Server(providers=[self.sprov], isolate=True)
        _log.debug('Server Conf: %s', self.server.conf())

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
        super(TestNewHandler, self).tearDown()

    def test_open(self):
        # Note that the mock.patch changes the ValueChangeHandler.open() into a noop
        with patch('p4p.test.test_sharedpv.TestNewHandler.ValueChangeHandler.open') as mock_open:
            # Test handler.open() calls processed correctly
            self.pv.open(1.0)
            self.assertEqual(self.pv.current(), 1.0)
            mock_open.assert_called_once()
            mock_open.reset_mock()

            # Test handler.open() calls processed correctly after close()
            self.pv.close()
            self.pv.open(2.0)
            self.assertEqual(self.pv.current(), 2.0)
            mock_open.assert_called_once()
            mock_open.reset_mock()

            # Test nothing goes wrong when we have no handler set in the SharedPV
            self.pv2.close()
            self.pv2.open(1.0)
            self.assertEqual(self.pv2.current(), 1.0)
            mock_open.assert_not_called()
            mock_open.reset_mock()

        # Check that value changes in a handler happen correctly
        self.pv.close()
        self.pv.open(55.0)
        self.assertEqual(self.pv.current(), 1.1)

        # Check that handler_open_ arguments are passed correctly
        self.pv.close()
        self.pv.open(33.0, handler_open_enable=False)
        self.assertEqual(self.pv.current(), 33.0)

    def test_post(self):
        # Note that the mock.patch changes the ValueChangeHandler.post() into a noop
        with patch('p4p.test.test_sharedpv.TestNewHandler.ValueChangeHandler.post') as mock_post:
            # Test handler.open() calls processed correctly; again patch means our function isn't called
            self.pv.open(1.0)
            self.pv.post(5.0)
            mock_post.assert_called_once()
            self.assertEqual(self.pv.current(), 5.0)
            mock_post.reset_mock()

            # Test nothing goes wrong when we have no handler set in the SharedPV
            self.pv2.post(6.0)
            mock_post.assert_not_called()
            self.assertEqual(self.pv2.current(), 6.0)
            mock_post.reset_mock()        

        # Check that value changes in a handler happen correctly
        self.pv.close()
        self.pv.open(1.0)
        self.pv.post(9.9)
        self.assertEqual(self.pv.current(), 2.2)

        # Check that handler_post_ arguments are passed correctly
        self.pv.close()
        self.pv.open(1.0)
        self.pv.post(77.0, handler_post_enable=False)
        self.assertEqual(self.pv.current(), 77.0)

    def test_get(self):
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:
            _log.debug('Client conf: %s', ctxt.conf())
            # PV not yet opened
            self.assertRaises(TimeoutError, ctxt.get, 'foo', timeout=0.1)

            self.pv.open(1.0)

            V = ctxt.get('foo')
            self.assertEqual(V, 1.1)
            self.assertTrue(V.raw.changed('value'))

            self.assertEqual(ctxt.get(['foo', 'bar']), [1.1, 42.0])

        C = weakref.ref(ctxt)
        del ctxt
        gc.collect()
        self.assertIsNone(C())

    def test_put_get(self):
        # Test handler.put() is called as expected and nothing breaks if it is absent 
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt, \
             patch('p4p.test.test_sharedpv.TestNewHandler.ValueChangeHandler.put') as mock_put:

            # Check the handler.put() is called
            self.pv.open(1.0)

            V = ctxt.get('foo')
            self.assertEqual(V, 1.1)

            # This will timeout as the patch has rendered the handler.post() a noop
            # and so we'll never issue an op.done() but we can still check it was called
            self.assertRaises(TimeoutError, ctxt.put, 'foo', 5, timeout=0.1)
            mock_put.assert_called_once()
            mock_put.reset_mock()

            # Check that the new code does not affect the operation of a SharedPV with no handler
            # This will also timeout as a PV without handler doesn't allow puts!
            V = ctxt.get('bar')
            self.assertEqual(V, 42.0)

            self.assertRaises(RemoteError, ctxt.put, 'bar', 5, timeout=0.1)
            mock_put.assert_not_called()

        C = weakref.ref(ctxt)
        del ctxt
        gc.collect()
        self.assertIsNone(C())

        # Test interaction with handler.post()
        self.pv.close()
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt, \
             patch('p4p.test.test_sharedpv.TestNewHandler.ValueChangeHandler.post') as mock_post:

            self.pv.open(1.0)
            V = ctxt.get('foo')
            self.assertEqual(V, 1.1)

            ctxt.put('foo', 15)

            self.assertEqual(ctxt.get('foo'), 3.3)
            mock_post.assert_called_once()

        C = weakref.ref(ctxt)
        del ctxt
        gc.collect()
        self.assertIsNone(C())

        # Test interaction with handler.post() without any patches to interfere
        self.pv.close()
        with Context('pva', conf=self.server.conf(), useenv=False) as ctxt:

            self.pv.open(1.0)
            V = ctxt.get('foo')
            self.assertEqual(V, 1.1)

            ctxt.put('foo', 15)

            self.assertEqual(ctxt.get('foo'), 2.2)
            mock_post.assert_called_once()

        C = weakref.ref(ctxt)
        del ctxt
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
                self.assertIsNone(ret)

    def test_rpc_error(self):
        with Server(providers=[self.provider], isolate=True) as S:
            with Context('pva', conf=S.conf(), useenv=False) as C:

                args = NTURI([
                    ('oops', '?'),
                ])

                with self.assertRaisesRegex(RemoteError, 'oops'):
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


#class TestPVRequestSlice(TestPVRequestMask):
#    mode = 'Slice'

class TestFirstLast(RefTestCase):
    maxDiff = 1000
    timeout = 1.0
    mode = 'Mask'

    class Handler:
        def __init__(self):
            self.evtC = threading.Event()
            self.evtD = threading.Event()
            self.conn = None
        def onFirstConnect(self, pv):
            _log.debug("onFirstConnect")
            self.conn = True
            self.evtC.set()
        def onLastDisconnect(self, pv):
            _log.debug("onLastDisconnect")
            self.conn = False
            self.evtD.set()

    def setUp(self):
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

                self.assertIsInstance(Q.get(timeout=self.timeout), Disconnected)
                self.assertIsInstance(Q.get(timeout=self.timeout), Value) # initial update

                _log.debug('TEST')
                self.H.evtC.wait(self.timeout) # onFirstConnect()
                self.assertTrue(self.H.conn)

        self.H.evtD.wait(self.timeout) # onLastDisconnect()
        _log.debug('SHUTDOWN')
        self.assertFalse(self.H.conn)

    def testServerShutdown(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
            Q = Queue(maxsize=4)
            with ctxt.monitor('foo', Q.put, notify_disconnect=True):

                Q.get(timeout=self.timeout) # initial update

                _log.debug('TEST')
                self.H.evtC.wait(self.timeout) # onFirstConnect()
                self.assertIs(self.H.conn, True)

                self.server.stop()

                self.H.evtD.wait(self.timeout) # onLastDisconnect()
                _log.debug('SHUTDOWN')
                self.assertIs(self.H.conn, False)

    def testPVClose(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
            Q = Queue(maxsize=4)
            with ctxt.monitor('foo', Q.put, notify_disconnect=True):

                Q.get(timeout=self.timeout) # initial update

                _log.debug('TEST')
                self.H.evtC.wait(self.timeout) # onFirstConnect()
                self.assertTrue(self.H.conn)

                self.pv.close(destroy=True, sync=True, timeout=self.timeout) # onLastDisconnect()

                _log.debug('CLOSE')
                self.assertFalse(self.H.conn)
