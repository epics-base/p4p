
import logging, warnings
_log = logging.getLogger(__name__)

import unittest, sys, random, weakref, gc, threading
from unittest.case import SkipTest

from ..nt import NTScalar
from ..server import Server, installProvider, removeProvider, StaticProvider
from .utils import RefTestCase

try:
    import asyncio
except ImportError:
    raise SkipTest('No asyncio')
else:
    from ..client.asyncio import Context, Disconnected
    from ..server.asyncio import SharedPV

    from .utils import inloop, clearloop

    class Handler:
        def put(self, pv, op):
            _log.debug("putting %s <- %s", op.name(), op.value())
            yield from asyncio.sleep(0, loop=self.loop) # prove that we can
            pv.post(op.value()*2)
            op.done()

    class TestGPM(RefTestCase):
        timeout = 5 # overall timeout for each test method

        @inloop
        @asyncio.coroutine
        def setUp(self):
            super(TestGPM, self).setUp()

            self.pv = SharedPV(nt=NTScalar('i'), initial=0, handler=Handler(), loop=self.loop)
            self.provider = StaticProvider('dut')
            self.provider.add('foo', self.pv)

        def tearDown(self):
            del self.pv
            del self.provider
            clearloop(self)
            super(TestGPM, self).tearDown()

        @inloop
        @asyncio.coroutine
        def test_getput(self):
            with Server(providers=[self.provider], isolate=True) as S:
                with Context('pva', conf=S.conf(), useenv=False) as C:
                    self.assertEqual(0, (yield from C.get('foo')))

                    yield from C.put('foo', 5)

                    self.assertEqual(5*2, (yield from C.get('foo')))

        @inloop
        @asyncio.coroutine
        def test_monitor(self):
            with Server(providers=[self.provider], isolate=True) as S:
                with Context('pva', conf=S.conf(), useenv=False) as C:

                    Q = asyncio.Queue(loop=self.loop)

                    sub = C.monitor('foo', Q.put, notify_disconnect=True)
                    try:
                        self.assertIsInstance((yield from Q.get()), Disconnected)

                        self.assertEqual(0, (yield from Q.get()))

                        yield from C.put('foo', 2)

                        self.assertEqual(2*2, (yield from Q.get()))

                        self.pv.close()

                        self.assertIsInstance((yield from Q.get()), Disconnected)

                        self.pv.open(3)

                        self.assertEqual(3, (yield from Q.get()))

                    finally:
                        sub.close()
                        yield from sub.wait_closed()
