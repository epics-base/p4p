import logging
import warnings
import platform
import unittest
import gc
import json
import weakref
import threading

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

try:
    from io import StringIO
except ImportError:
    from cString import StringIO

from .utils import RefTestCase, RegularNamedTemporaryFile as NamedTemporaryFile
from ..server import Server, StaticProvider, removeProvider
from ..server.thread import SharedPV, _defaultWorkQueue
from ..client.thread import Context, Disconnected, TimeoutError, RemoteError
from ..nt import NTScalar
from ..gw import App, main, getargs

from .. import _gw

_log = logging.getLogger(__name__)

class TestGC(RefTestCase):
    def test_empty(self):
        class Dummy(object):
            pass
        H = Dummy()
        CLI = _gw.Client(u'pva', {'EPICS_PVA_BROADCAST_PORT': '0',
                                  'EPICS_PVA_SERVER_PORT': '0',
                                  'EPICS_PVAS_INTF_ADDR_LIST': '127.0.0.1',
                                  'EPICS_PVA_ADDR_LIST': '127.0.0.1',
                                  'EPICS_PVA_AUTO_ADDR_LIST': '0'})
        GW = _gw.Provider(u'nulltest', CLI, H)
        removeProvider(u'nulltest')

        self.assertEqual(GW.use_count(), 1)

        h = weakref.ref(H)
        gw = weakref.ref(GW)
        del H
        del GW
        del CLI
        gc.collect()

        self.assertIsNone(h())
        self.assertIsNone(gw())

    def test_server(self):
        class Dummy(object):
            pass
        H = Dummy()
        CLI = _gw.Client(u'pva', {'EPICS_PVA_BROADCAST_PORT': '0',
                                  'EPICS_PVA_SERVER_PORT': '0',
                                  'EPICS_PVAS_INTF_ADDR_LIST': '127.0.0.1',
                                  'EPICS_PVA_ADDR_LIST': '127.0.0.1',
                                  'EPICS_PVA_AUTO_ADDR_LIST': '0'})
        GW = _gw.Provider(u'nulltest', CLI, H)

        try:
            with Server(providers=['nulltest'], isolate=True):
                self.assertFalse(GW.testChannel(b'invalid:pv:name'))
        finally:
            removeProvider(u'nulltest')

        h = weakref.ref(H)
        gw = weakref.ref(GW)
        del H
        del GW
        del CLI
        gc.collect()

        self.assertIsNone(h())
        self.assertIsNone(gw())

class TestLowLevel(RefTestCase):
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

        def audit(self, msg):
            _log.info("AUDIT: %s", msg)

    def setUp(self):
        super(TestLowLevel, self).setUp()

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
        CLI = _gw.Client(u'pva', self._us_server.conf())
        H.provider = self.gw = _gw.Provider(u'gateway', CLI, H)

        try:
            # GW server side
            self._ds_server = Server(providers=['gateway'], isolate=True)
        finally:
            # don't need this in the global registry anymore.
            # Server holds strong ref.
            removeProvider(u'gateway')

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

        super(TestLowLevel, self).tearDown()

    def test_get(self):
        val = self._ds_client.get('pv:ro', timeout=self.timeout)
        self.assertEqual(val, 42)
        self.pv.post(43)
        # will be delayed by holdoff logic
        val = self._ds_client.get('pv:ro', timeout=self.timeout)
        self.assertEqual(val, 43)

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

class TestApp(App):
    def __init__(self, args):
        super(TestApp, self).__init__(args)
        self.__evt = threading.Event()

    def abort(self):
        self.__evt.set()

    def sleep(self, dly):
        if self.__evt.wait(dly):
            raise KeyboardInterrupt

class TestHighLevel(RefTestCase):
    timeout = 10

    def setUp(self):
        _log.debug("Enter setUp")
        super(TestHighLevel, self).setUp()

        # upstream server
        self.pv = SharedPV(nt=NTScalar('i'), initial=42)
        self._us_provider = StaticProvider('upstream')
        self._us_provider.add('pv:name', self.pv)

        @self.pv.put
        def put(pv, op):
            _log.debug("PUT %s", op.value())
            pv.post(op.value())
            op.done()

        self._us_server = self._us_conf = None
        self.startServer()
        _log.debug("US server conf: %s", self._us_conf)
        self.assertNotEqual(0, self._us_conf['EPICS_PVA_BROADCAST_PORT'])

        cfile = self._cfile = NamedTemporaryFile('w+')
        json.dump({
            'version':2,
            'clients':[{
                'name':'client1',
                'provider':'pva',
                'addrlist':'127.0.0.1',
                'autoaddrlist':False,
                'bcastport':self._us_conf['EPICS_PVA_BROADCAST_PORT'],
                'serverport':0,
            }],
            'servers':[{
                'name':'server1',
                'clients':['client1'],
                'interface':['127.0.0.1'],
                'addrlist':'127.0.0.1',
                'autoaddrlist':False,
                'bcastport':0,
                'serverport':0,
            }],
        }, cfile)
        cfile.flush()
        with open(cfile.name, 'r') as F:
            _log.debug('GW config')
            _log.debug(F.read())

        # gateway
        args = getargs().parse_args(['--no-ban-local', '-v', cfile.name])

        self._app = TestApp(args)
        self._main = threading.Thread(target=self._app.run, name='GW Main')
        _log.debug("DS server conf: %s", self._app.servers[u'server1'].conf())

        # downstream client
        self._ds_client = Context('pva', conf=self._app.servers[u'server1'].conf(), useenv=False)

        self._main.start()
        _log.debug("Exit setUp")

    def startServer(self):
        if self._us_server is None:
            if self._us_conf is None:
                _log.debug("Starting server fresh")
                self._us_server = Server(providers=[self._us_provider], isolate=True)
                self._us_conf = self._us_server.conf()
            else:
                _log.debug("Starting server with %s", self._us_conf)
                self._us_server = Server(providers=[self._us_provider], conf=self._us_conf, useenv=False)

    def stopServer(self):
        if self._us_server is not None:
            _log.debug("Stopping server")
            self._us_server.stop()
            self._us_server = None

    def tearDown(self):
        _log.debug("Enter tearDown")
        # downstream client
        self._ds_client.close()
        del self._ds_client

        # gateway
        self._app.abort()
        self._main.join(self.timeout)
        del self._app
        del self._main

        # upstream server
        self.stopServer()
        del self._us_provider
        del self._us_server
        del self.pv
        _defaultWorkQueue.sync()

        super(TestHighLevel, self).tearDown()
        _log.debug("Exit tearDown")

    def test_get(self):
        val = self._ds_client.get('pv:name', timeout=self.timeout)
        self.assertEqual(val, 42)
        self.assertTrue(val.raw.changed('value'))
        # re-read right away to check throttling/holdoff logic
        self.assertEqual(val, 42)

    def test_get_mask(self):
        val = self._ds_client.get('pv:name', timeout=self.timeout, request='timeStamp')
        self.assertEqual(val, 0) # not requested at default
        self.assertFalse(val.raw.changed('value'))

    def test_get_bad_mask(self):
        with self.assertRaisesRegexp(RemoteError, "No field 'nonexistant' Empty field selection"):
            val = self._ds_client.get('pv:name', timeout=self.timeout, request='nonexistant')

    def test_put(self):
        self._ds_client.put('pv:name', 41, timeout=self.timeout)

        val = self._ds_client.get('pv:name', timeout=self.timeout)
        self.assertEqual(val, 41)

    def test_put_bad_mask(self):
        with self.assertRaisesRegexp(RemoteError, "No field 'nonexistant' Empty field selection"):
            self._ds_client.put('pv:name', 41, request='nonexistant', timeout=self.timeout)

    def test_mon(self):
        """Setup a monitor through the GW
        """
        Q = Queue(maxsize=4)
        with self._ds_client.monitor('pv:name', Q.put, notify_disconnect=True):
            V = Q.get(timeout=self.timeout)
            self.assertIsInstance(V, Disconnected)

            _log.debug("Wait for initial update")
            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 42)

    def test_mon_disconn(self):
        """See that upstream disconnect propagates
        """
        Q = Queue(maxsize=4)
        with self._ds_client.monitor('pv:name', Q.put, notify_disconnect=True):
            V = Q.get(timeout=self.timeout)
            self.assertIsInstance(V, Disconnected)

            with self.assertRaises(Empty):
                Q.get(timeout=0.1)

            _log.debug("Wait for initial update")
            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 42)

            self.stopServer()

            _log.debug("Wait for Disconnected")
            V = Q.get(timeout=self.timeout)
            self.assertIsInstance(V, Disconnected)

    @unittest.skipIf(platform.system()=='Windows',
                     "not responding to searches after restart.  wrong IP?")
    def test_mon_disconn_reconn(self):
        """Start disconnected, the connect and disconnect
        """
        self.stopServer()

        Q = Queue(maxsize=4)
        with self._ds_client.monitor('pv:name', Q.put, notify_disconnect=True):
            V = Q.get(timeout=self.timeout)
            self.assertIsInstance(V, Disconnected)

            with self.assertRaises(Empty):
                Q.get(timeout=0.1)

            self.startServer()

            _log.debug("Wait for initial update")
            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 42)

            self.stopServer()

            _log.debug("Wait for Disconnected")
            V = Q.get(timeout=self.timeout)
            self.assertIsInstance(V, Disconnected)

            _log.debug("after Disconnected")
            self.pv.post(5)

            with self.assertRaises(Empty):
                Q.get(timeout=0.1)

            self.startServer()

            _log.debug("Wait for reconnect update")
            V = Q.get(timeout=self.timeout)
            self.assertEqual(V, 5)

class TestTestServer(RefTestCase):
    conf_template = '''
{
    "version":2,
    "clients":[
        {
            "name":"aclient",
            "provider":"pva",
            "addrlist":"1.2.3.4",
            "autoaddrlist":false,
            "serverport":5085,
            "bcastport":5086
        }
    ],
    "servers":[
        {
            "name":"theserver",
            "clients":["aclient"],
            "interface":["4.3.2.1"],
            "addrlist":"",
            "autoaddrlist":false,
            "serverport":5075,
            "bcastport":5076,
            "getholdoff":1.0,
            "statusprefix":"sts:",
            "access":"%(acf)s",
            "pvlist":"%(pvlist)s",
            "acf_client":"aclient"
        }
    ]
}
'''

    def setUp(self):
        RefTestCase.setUp(self)
        self._files = []
        self._log = StringIO()
        self._handler = logging.StreamHandler(self._log)
        logging.getLogger().addHandler(self._handler)

    def tearDown(self):
        [f.close() for f in self._files]
        logging.getLogger().removeHandler(self._handler)
        RefTestCase.tearDown(self)

    def log(self):
        self._handler.flush()
        self._log.seek(0)
        return self._log.read()

    def write(self, content):
        F = NamedTemporaryFile('w+')
        self._files.append(F)
        F.write(content)
        F.flush()
        return F.name

    def test_ok(self):
        acf = self.write('''
    ASG(DEFAULT) {
        RULE(1, WRITE)
        RULE(1, UNCACHED)
    }
''')
        pvlist = self.write('''
.* ALLOW
''')

        conf = self.write(self.conf_template%{'acf':repr(acf)[1:-1], 'pvlist':repr(pvlist)[1:-1]})

        main(['-T', conf])

    def test_bad_acf(self):
        acf = self.write('''
    ASG(DEFAULT)
        RULE(1, WRITE)
        RULE(1, UNCACHED)
    }
''')
        pvlist = self.write('''
.* ALLOW
''')

        conf = self.write(self.conf_template%{'acf':repr(acf)[1:-1], 'pvlist':repr(pvlist)[1:-1]})

        with self.assertRaises(SystemExit):
            main(['-T', conf])

        self.assertRegex(self.log(), r".*Syntax error.*RULE.*")

    def test_bad_pvlist(self):
        acf = self.write('''
    ASG(DEFAULT) {
        RULE(1, WRITE)
        RULE(1, UNCACHED)
    }
''')
        pvlist = self.write('''
.* ALLW
''')

        conf = self.write(self.conf_template%{'acf':repr(acf)[1:-1], 'pvlist':repr(pvlist)[1:-1]})

        with self.assertRaises(SystemExit):
            main(['-T', conf])

        self.assertRegex(self.log(), r".*Unknown command.*ALLW.*")
