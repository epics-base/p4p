
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

from ..nt import NTScalar
from ..server import Server, StaticProvider
from .utils import RefTestCase

try:
    import asyncio
except ImportError:
    raise SkipTest('No asyncio')
    # not that this is going to help as 'yield from' is a syntax error in 2.7
else:
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
            _log.debug("put complete %s <- %s", op.name(), op.value())

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

        @inloop
        @asyncio.coroutine
        def cleanup(self):
            yield from self.pv.shutdown()
            yield from self.pv2.shutdown()

        def tearDown(self):
            self.cleanup()
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
