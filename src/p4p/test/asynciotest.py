# I'm imported from test_asyncio.y
# Add new TestCase s to __all__

import logging
_log = logging.getLogger(__name__)

from functools import wraps

from .. import Value
from ..nt import NTScalar
from ..server import Server, StaticProvider
from .utils import RefTestCase

import asyncio

from ..client.asyncio import Context, Disconnected, timesout
from ..server.asyncio import SharedPV

__all__ = (
    'TestGPM',
    'TestTimeout',
    'TestFirstLast',
)

# we should never implicitly use the default loop.
asyncio.get_event_loop().close()

class AsyncMeta(type):
    """Automatically wrap and "async def test*():" methods for dispatch to self.loop
    """
    def __new__(klass, name, bases, classdict):
        for name, mem in classdict.items():
            if name.startswith('test') and asyncio.iscoroutinefunction(mem):
                @wraps(mem)
                def wrapper(self, mem=mem):
                    self.loop.run_until_complete(asyncio.wait_for(mem(self), self.timeout))
                classdict[name] = wrapper

        return type.__new__(klass, name, bases, classdict)

class AsyncTest(RefTestCase, metaclass=AsyncMeta):
    timeout = 1.0

    def setUp(self):
        super(AsyncTest, self).setUp()
        self.loop = asyncio.new_event_loop()
        self.loop.set_debug(True)
        self.loop.run_until_complete(asyncio.wait_for(self.asyncSetUp(), self.timeout))

    def tearDown(self):
        self.loop.run_until_complete(asyncio.wait_for(self.asyncTearDown(), self.timeout))
        self.loop.close()
        super(AsyncTest, self).tearDown()

    async def asyncSetUp(self):
        pass

    async def asyncTearDown(self):
        pass

class TestGPM(AsyncTest):
    timeout = 3  # overall timeout for each test method

    class Handler:
        async def put(self, pv, op):
            _log.debug("putting %s <- %s", op.name(), op.value())
            await asyncio.sleep(0)  # prove that we can
            pv.post(op.value() * 2)
            op.done()

    async def asyncSetUp(self):
        await super(TestGPM, self).asyncSetUp()

        self.pv = SharedPV(nt=NTScalar('i'), initial=0, handler=self.Handler())
        self.pv2 = SharedPV(handler=self.Handler(), nt=NTScalar('d'), initial=42.0)
        self.provider = StaticProvider("serverend")
        self.provider.add('foo', self.pv)
        self.provider.add('bar', self.pv2)

    async def asyncTearDown(self):
        del self.pv
        del self.pv2
        del self.provider
        await super(TestGPM, self).asyncTearDown()

    async def test_getput(self):
        with Server(providers=[self.provider], isolate=True) as S:
            with Context('pva', conf=S.conf(), useenv=False) as C:
                self.assertEqual(0, (await C.get('foo')))

                await C.put('foo', 5)

                self.assertEqual(5 * 2, (await C.get('foo')))

    async def test_monitor(self):
        with Server(providers=[self.provider], isolate=True) as S:
            with Context('pva', conf=S.conf(), useenv=False) as C:

                Q = asyncio.Queue()

                sub = C.monitor('foo', Q.put, notify_disconnect=True)
                try:
                    self.assertIsInstance((await Q.get()), Disconnected)

                    self.assertEqual(0, (await Q.get()))

                    await C.put('foo', 2)

                    self.assertEqual(2 * 2, (await Q.get()))

                    self.pv.close()

                    self.assertIsInstance((await Q.get()), Disconnected)

                    self.pv.open(3)

                    self.assertEqual(3, (await Q.get()))

                finally:
                    sub.close()
                    await sub.wait_closed()

    async def test_put_noconvert(self):
        with Server(providers=[self.provider], isolate=True) as S:
            with Context('pva', conf=S.conf(), useenv=False) as C:
                with self.assertRaisesRegex(ValueError, 'not_an_integer'):
                    await C.put('foo', 'not_an_integer')

    if hasattr(asyncio, 'coroutine'):
        @asyncio.coroutine
        def test_gen_coro(self):
            # demonstrate interoperability w/ code using generated based coroutines
            with Server(providers=[self.provider], isolate=True) as S:
                with Context('pva', conf=S.conf(), useenv=False) as C:
                    self.assertEqual(0, (yield from C.get('foo')))

                    yield from C.put('foo', 5)

                    self.assertEqual(5 * 2, (yield from C.get('foo')))


class TestTimeout(AsyncTest):

    async def test_timeout(self):
        done = None

        @timesout()
        async def action():
            nonlocal done
            done = False
            await asyncio.sleep(5)
            done = True

        with self.assertRaises(asyncio.TimeoutError):
            await action(timeout=0.01)

class TestFirstLast(AsyncTest):
    maxDiff = 2000
    timeout = 5.0
    mode = 'Mask'

    class Handler:
        def __init__(self):
            self.evtC = asyncio.Event()
            self.evtD = asyncio.Event()
            self.conn = None
        def onFirstConnect(self, pv):
            _log.debug("onFirstConnect")
            self.conn = True
            self.evtC.set()
        def onLastDisconnect(self, pv):
            _log.debug("onLastDisconnect")
            self.conn = False
            self.evtD.set()

    async def asyncSetUp(self):
        await super(TestFirstLast, self).asyncSetUp()

        self.H = self.Handler()
        self.pv = SharedPV(handler=self.H,
                           nt=NTScalar('d'),
                           options={'mapperMode':self.mode})
        self.sprov = StaticProvider("serverend")
        self.sprov.add('foo', self.pv)

        self.server = Server(providers=[self.sprov], isolate=True)

    async def asyncTearDown(self):
        self.server.stop()
        #_defaultWorkQueue.stop()
        del self.server
        del self.sprov
        del self.pv
        del self.H
        await super(TestFirstLast, self).asyncTearDown()

    async def testClientDisconn(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
            Q = asyncio.Queue(maxsize=4)
            sub = ctxt.monitor('foo', Q.put, notify_disconnect=True)
            try:

                _log.debug('Wait Disconnected')
                E = await Q.get() # Disconnected
                self.assertIsInstance(E, Disconnected)
                _log.debug('Wait initial')
                E = await Q.get() # initial update
                self.assertIsInstance(E, Value)

                _log.debug('Wait onFirstConnect')
                await self.H.evtC.wait() # onFirstConnect()
                self.assertTrue(self.H.conn)

            except:
                _log.exception('oops')
                raise
            finally:
                _log.debug('sub close()')
                sub.close()
                await sub.wait_closed()

        _log.debug('Wait onLastDisconnect')
        await self.H.evtD.wait() # onLastDisconnect()
        _log.debug('SHUTDOWN')
        self.assertFalse(self.H.conn)

    async def testServerShutdown(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
            Q = asyncio.Queue(maxsize=4)
            sub = ctxt.monitor('foo', Q.put, notify_disconnect=True)
            try:

                await Q.get() # initial update

                _log.debug('TEST')
                await self.H.evtC.wait() # onFirstConnect()
                self.assertIs(self.H.conn, True)

                self.server.stop()

                await self.H.evtD.wait() # onLastDisconnect()
                _log.debug('SHUTDOWN')
                self.assertIs(self.H.conn, False)

            finally:
                sub.close()
                await sub.wait_closed()

    async def testPVClose(self):
        self.pv.open(1.0)

        with Context('pva', conf=self.server.conf(), useenv=False, unwrap={}) as ctxt:
            Q = asyncio.Queue(maxsize=4)
            sub = ctxt.monitor('foo', Q.put, notify_disconnect=True)
            try:

                await Q.get() # initial update

                _log.debug('TEST')
                await self.H.evtC.wait() # onFirstConnect()
                self.assertTrue(self.H.conn)

                await self.pv.close(destroy=True, sync=True) # onLastDisconnect()

                _log.debug('CLOSE')
                self.assertFalse(self.H.conn)

            finally:
                sub.close()
                await sub.wait_closed()
