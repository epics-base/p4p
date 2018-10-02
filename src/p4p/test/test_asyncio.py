
import logging
import warnings
_log = logging.getLogger(__name__)

import unittest
import sys
import random
import weakref
import gc
from unittest.case import SkipTest

from ..nt import NTScalar
from ..server import Server, StaticProvider
from .utils import RefTestCase

import asyncio

from ..client.asyncio import Context, Disconnected, timesout
from ..server.asyncio import SharedPV

from .utils import inloop, clearloop

class Handler:

    @asyncio.coroutine
    def put(self, pv, op):
        _log.debug("putting %s <- %s", op.name(), op.value())
        yield from asyncio.sleep(0, loop=self.loop)  # prove that we can
        pv.post(op.value() * 2)
        op.done()

class TestGPM(RefTestCase):
    timeout = 3  # overall timeout for each test method

    @inloop
    @asyncio.coroutine
    def setUp(self):
        super(TestGPM, self).setUp()

        self.pv = SharedPV(nt=NTScalar('i'), initial=0, handler=Handler(), loop=self.loop)
        self.pv2 = SharedPV(handler=Handler(), nt=NTScalar('d'), initial=42.0, loop=self.loop)
        self.provider = StaticProvider("serverend")
        self.provider.add('foo', self.pv)
        self.provider.add('bar', self.pv2)

    def tearDown(self):
        del self.pv
        del self.pv2
        del self.provider
        clearloop(self)
        super(TestGPM, self).tearDown()

    @inloop
    @asyncio.coroutine
    def test_getput(self):
        with Server(providers=[self.provider], isolate=True) as S:
            with Context('pva', conf=S.conf(), useenv=False, loop=self.loop) as C:
                self.assertEqual(0, (yield from C.get('foo')))

                yield from C.put('foo', 5)

                self.assertEqual(5 * 2, (yield from C.get('foo')))

    @inloop
    @asyncio.coroutine
    def test_monitor(self):
        with Server(providers=[self.provider], isolate=True) as S:
            with Context('pva', conf=S.conf(), useenv=False, loop=self.loop) as C:

                Q = asyncio.Queue(loop=self.loop)

                sub = C.monitor('foo', Q.put, notify_disconnect=True)
                try:
                    self.assertIsInstance((yield from Q.get()), Disconnected)

                    self.assertEqual(0, (yield from Q.get()))

                    yield from C.put('foo', 2)

                    self.assertEqual(2 * 2, (yield from Q.get()))

                    self.pv.close()

                    self.assertIsInstance((yield from Q.get()), Disconnected)

                    self.pv.open(3)

                    self.assertEqual(3, (yield from Q.get()))

                finally:
                    sub.close()
                    yield from sub.wait_closed()


class TestTimeout(unittest.TestCase):

    def tearDown(self):
        clearloop(self)

    @inloop
    def test_timeout(self):
        done = None

        if sys.version_info >= (3, 4) and sys.version_info < (3, 5):
            raise SkipTest("wait_for() kind of broken in 3.4")
            # I'm seeing a test failure with "got Future <Future pending> attached to a different loop"
            # but I can't find where the different loop gets mixed in.
            # So I _think_ this is a bug in 3.4.

        @timesout()
        @asyncio.coroutine
        def action(loop):
            nonlocal done
            done = False
            yield from asyncio.sleep(5, loop=loop)
            done = True

        try:
            yield from action(self.loop, timeout=0.01)
            self.assertTrue(False)
        except asyncio.TimeoutError:
            pass

        self.assertIs(done, False)

class TestFirstLast(RefTestCase):
    maxDiff = 1000
    timeout = 1.0
    mode = 'Mask'

    class Handler:
        def __init__(self, loop):
            self.evt = asyncio.Event(loop=loop)
            self.conn = None
        def onFirstConnect(self, pv):
            _log.debug("onFirstConnect")
            self.conn = True
            self.evt.set()
        def onLastDisconnect(self, pv):
            _log.debug("onLastDisconnect")
            self.conn = False
            self.evt.set()

    @inloop
    @asyncio.coroutine
    def setUp(self):
        # gc.set_debug(gc.DEBUG_LEAK)
        super(TestFirstLast, self).setUp()

        self.H = self.Handler(self.loop)
        self.pv = SharedPV(handler=self.H,
                           nt=NTScalar('d'),
                           options={'mapperMode':self.mode},
                           loop=self.loop)
        self.sprov = StaticProvider("serverend")
        self.sprov.add('foo', self.pv)

        self.server = Server(providers=[self.sprov], isolate=True)

    def tearDown(self):
        self.server.stop()
        #_defaultWorkQueue.stop()
        del self.server
        del self.sprov
        del self.pv
        del self.H
        clearloop(self)
        super(TestFirstLast, self).tearDown()

    @inloop
    @asyncio.coroutine
    def testClientDisconn(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}, loop=self.loop) as ctxt:
            Q = asyncio.Queue(maxsize=4, loop=self.loop)
            sub = ctxt.monitor('foo', Q.put, notify_disconnect=True)
            try:

                yield from Q.get() # initial update

                _log.debug('TEST')
                yield from self.H.evt.wait() # onFirstConnect()
                self.H.evt.clear()
                self.assertTrue(self.H.conn)

            finally:
                sub.close()
                yield from sub.wait_closed()

        yield from self.H.evt.wait() # onLastDisconnect()
        _log.debug('SHUTDOWN')
        self.assertFalse(self.H.conn)

    @inloop
    @asyncio.coroutine
    def testServerShutdown(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}, loop=self.loop) as ctxt:
            Q = asyncio.Queue(maxsize=4, loop=self.loop)
            sub = ctxt.monitor('foo', Q.put, notify_disconnect=True)
            try:

                yield from Q.get() # initial update

                _log.debug('TEST')
                yield from self.H.evt.wait() # onFirstConnect()
                self.H.evt.clear()
                self.assertIs(self.H.conn, True)

                self.server.stop()

                yield from self.H.evt.wait() # onLastDisconnect()
                _log.debug('SHUTDOWN')
                self.assertIs(self.H.conn, False)

            finally:
                sub.close()
                yield from sub.wait_closed()

    @inloop
    @asyncio.coroutine
    def testPVClose(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}, loop=self.loop) as ctxt:
            Q = asyncio.Queue(maxsize=4, loop=self.loop)
            sub = ctxt.monitor('foo', Q.put, notify_disconnect=True)
            try:

                yield from Q.get() # initial update

                _log.debug('TEST')
                yield from self.H.evt.wait() # onFirstConnect()
                self.H.evt.clear()
                self.assertTrue(self.H.conn)

                yield from self.pv.close(destroy=True, sync=True) # onLastDisconnect()

                _log.debug('CLOSE')
                self.assertFalse(self.H.conn)

            finally:
                sub.close()
                yield from sub.wait_closed()
