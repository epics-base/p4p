import logging
import warnings

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

import unittest
import gc
import weakref

from .utils import RefTestCase
from ..server import Server, StaticProvider, removeProvider
from ..server.thread import SharedPV, _defaultWorkQueue
from ..client.thread import Context, Disconnected, TimeoutError, RemoteError
from ..nt import NTScalar

from .. import _gw

_log = logging.getLogger(__name__)

class TestGC(RefTestCase):
    def test_empty(self):
        class Dummy(object):
            pass
        H = Dummy()
        GW = _gw.installGW('nulltest', {'EPICS_PVA_BROADCAST_PORT': '0',
                                        'EPICS_PVA_SERVER_PORT': '0',
                                        'EPICS_PVAS_INTF_ADDR_LIST': '127.0.0.1',
                                        'EPICS_PVA_ADDR_LIST': '127.0.0.1',
                                        'EPICS_PVA_AUTO_ADDR_LIST': '0'}, H)
        removeProvider('nulltest')

        self.assertEqual(GW.use_count(), 1)

        h = weakref.ref(H)
        gw = weakref.ref(GW)
        del H
        del GW
        gc.collect()

        self.assertIsNone(h())
        self.assertIsNone(gw())

    def test_server(self):
        class Dummy(object):
            pass
        H = Dummy()
        GW = _gw.installGW('nulltest', {'EPICS_PVA_BROADCAST_PORT': '0',
                                        'EPICS_PVA_SERVER_PORT': '0',
                                        'EPICS_PVAS_INTF_ADDR_LIST': '127.0.0.1',
                                        'EPICS_PVA_ADDR_LIST': '127.0.0.1',
                                        'EPICS_PVA_AUTO_ADDR_LIST': '0'}, H)

        try:
            with Server(providers=['nulltest'], isolate=True):
                self.assertFalse(GW.testChannel(b'invalid:pv:name'))
        finally:
            removeProvider('nulltest')

        h = weakref.ref(H)
        gw = weakref.ref(GW)
        del H
        del GW
        gc.collect()

        self.assertIsNone(h())
        self.assertIsNone(gw())

class TestGW(RefTestCase):
    timeout = 1

    class Handler(object):
        def testChannel(self, pvname, peer):
            # our permissions check
            if pvname in (b'pv:ro', b'pv:rw'):
                # Add to, and test, channel cache.  Return True if client channel is available
                ret = self.provider.testChannel(b'pv:name')
            else:
                ret = self.provider.BanPV
            _log.debug("GW Search %s from %s -> %s", pvname, peer, ret)
            return ret

        def makeChannel(self, op):
            try:
                # try to create from cache.  Does not add
                chan = op.create(b'pv:name')
                put = False
                if op.name==b'pv:rw':
                    put = True
                    chan.access(put=put, rpc=False, uncached=False)
                _log.debug("GW Create %s put=%s %s for %s of %s", op.name, put, chan, op.account, op.peer)
                return chan
            except:
                _log.exception("Unable to create channel for %s", op.name)

    def setUp(self):
        super(TestGW, self).setUp()

        # upstream server
        self.pv = SharedPV(nt=NTScalar('i'), initial=42)
        self._us_provider = StaticProvider('upstream')
        self._us_provider.add('pv:name', self.pv)

        @self.pv.put
        def put(pv, op):
            _log.debug("PUT %s", op.value())
            pv.post(op.value())
            op.done()

        self._us_server = Server(providers=[self._us_provider], isolate=True)

        # GW client side
        # placed weakref in global registry
        H = self.Handler()
        H.provider = self.gw = _gw.installGW('gateway', self._us_server.conf(), H)

        try:
            # GW server side
            self._ds_server = Server(providers=['gateway'], isolate=True)
        finally:
            # don't need this in the global registry anymore.
            # Server holds strong ref.
            removeProvider('gateway')

        # downstream client
        self._ds_client = Context('pva', conf=self._ds_server.conf(), useenv=False)

    def tearDown(self):
        self._ds_client.close()
        del self._ds_client

        self._ds_server.stop()
        del self._ds_server

        self._us_server.stop()
        del self._us_provider
        del self._us_server
        del self.pv
        _defaultWorkQueue.sync()
        gc.collect()

        self.assertEqual(self.gw.use_count(), 1)
        gw = weakref.ref(self.gw)
        del self.gw
        gc.collect()

        self.assertIsNone(gw())

        super(TestGW, self).tearDown()

    def test_get(self):
        val = self._ds_client.get('pv:ro', timeout=self.timeout)
        self.assertEqual(val, 42)

    def test_ban(self):
        with self.assertRaises(TimeoutError):
            self._ds_client.put('invalid', 40, timeout=0.1)
        # TODO: test cache

    def test_put(self):
        with self.assertRaises(RemoteError):
            self._ds_client.put('pv:ro', 40, timeout=self.timeout)

        self.assertEqual(self._ds_client.get('pv:ro', timeout=self.timeout), 42)

        self._ds_client.put('pv:rw', 41, timeout=self.timeout)

        self.assertEqual(self._ds_client.get('pv:ro', timeout=self.timeout), 41)
        self.assertEqual(self._ds_client.get('pv:rw', timeout=self.timeout), 41)

    def test_monitor(self):
        Q1 = Queue(maxsize=4)
        Q2 = Queue(maxsize=4)

        with self._ds_client.monitor('pv:ro', Q1.put, notify_disconnect=True):
            
            self.assertIsInstance(Q1.get(timeout=self.timeout), Disconnected)
            self.assertEqual(42, Q1.get(timeout=self.timeout))

            with self._ds_client.monitor('pv:rw', Q2.put, notify_disconnect=True):
                self.assertIsInstance(Q2.get(timeout=self.timeout), Disconnected)
                self.assertEqual(42, Q2.get(timeout=self.timeout))

                # no activity on first subscription
                with self.assertRaises(Empty):
                    Q2.get(timeout=0.01)
