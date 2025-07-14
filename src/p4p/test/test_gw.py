import logging
import warnings
import platform
import unittest
import gc
import json
import weakref
import threading

try: # 2.7
    from Queue import Queue, Full, Empty
    from time import time as monotonic
except ImportError: # 3.x
    from queue import Queue, Full, Empty
    from time import monotonic

try:
    from io import StringIO
except ImportError:
    from cString import StringIO

from .utils import RefTestCase, RegularNamedTemporaryFile as NamedTemporaryFile
from ..server import Server, StaticProvider
from ..server.thread import SharedPV, _defaultWorkQueue
from ..client import raw
from ..client.thread import Context, Disconnected, TimeoutError, RemoteError
from ..nt import NTScalar
from ..gw import App, main, getargs

from .. import _gw

_log = logging.getLogger(__name__)

class TestTemplate(unittest.TestCase):
    def test_config(self):
        with NamedTemporaryFile() as F:
            try:
                main(['--example-config', F.name])
            except SystemExit as e:
                self.assertEqual(e.code, 0)

            F.seek(0)
            content = F.read()
            self.assertRegex(content, '"statusprefix"')

    def test_systemd_instance(self):
        with NamedTemporaryFile(suffix='@blah.service') as F:
            try:
                main(['--example-systemd', F.name])
            except SystemExit as e:
                self.assertEqual(e.code, 0)

            F.seek(0)
            content = F.read()
            self.assertRegex(content, '-m p4p.gw /etc/pvagw/blah.conf')
            self.assertRegex(content, 'multi-user.target')

    def test_systemd_template(self):
        with NamedTemporaryFile(suffix='@.service') as F:
            try:
                main(['--example-systemd', F.name])
            except SystemExit as e:
                self.assertEqual(e.code, 0)

            F.seek(0)
            content = F.read()
            self.assertRegex(content, '-m p4p.gw /etc/pvagw/%i.conf')
            self.assertRegex(content, 'multi-user.target')

class TestGC(RefTestCase):
    def test_empty(self):
        class Dummy(object):
            pass
        H = Dummy()
        CLI = raw.Context(conf={'EPICS_PVA_BROADCAST_PORT': '0',
                                'EPICS_PVA_SERVER_PORT': '0',
                                'EPICS_PVAS_INTF_ADDR_LIST': '127.0.0.1',
                                'EPICS_PVA_ADDR_LIST': '127.0.0.1',
                                'EPICS_PVA_AUTO_ADDR_LIST': '0'})
        GW = _gw.Provider(u'nulltest', CLI, H)

        self.assertEqual(GW.use_count(), 2) # one for Provider, one for Source base class

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
        CLI = raw.Context(conf={'EPICS_PVA_BROADCAST_PORT': '0',
                                'EPICS_PVA_SERVER_PORT': '0',
                                'EPICS_PVAS_INTF_ADDR_LIST': '127.0.0.1',
                                'EPICS_PVA_ADDR_LIST': '127.0.0.1',
                                'EPICS_PVA_AUTO_ADDR_LIST': '0'})
        GW = _gw.Provider(u'nulltest', CLI, H)

        with Server(providers=[GW], isolate=True):
            self.assertFalse(GW.testChannel(b'invalid:pv:name'))

        h = weakref.ref(H)
        gw = weakref.ref(GW)
        del H
        del GW
        del CLI
        gc.collect()

        self.assertIsNone(h())
        self.assertIsNone(gw())

class TestLowLevel(RefTestCase):
    timeout = 5

    class Handler(object):
        def testChannel(self, pvname, peer):
            # our permissions check
            if pvname in (b'pv:ro', b'pv:rw'):
                # Add to, and test, channel cache.  Return True if client channel is available
                ret = self.provider.testChannel(b'pv:name')
            else:
                ret = self.provider.BanPV
            _log.debug("GW Search %r from %r -> %s", pvname, peer, ret)
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
        CLI = raw.Context(u'pva', self._us_server.conf())
        H.provider = self.gw = _gw.Provider(u'gateway', CLI, H)

        # GW server side
        self._ds_server = Server(providers=[H.provider], isolate=True)

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

        self.assertEqual(self.gw.use_count(), 2)
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

                # check stats now that we have an active connection
                S = self.gw.report(1.0)
                self.assertEqual(len(S), 1, S)
                self.assertEqual(S[0][0], "pv:name")

                S = _gw.Server_report(self._ds_server._S, 1.0)
                self.assertEqual(len(S), 2, S)
                self.assertEqual(S[0][0], "pv:name")
                self.assertEqual(S[1][0], "pv:name")
                # TODO: is order stable?
                self.assertEqual(S[0][1], "pv:ro")
                self.assertEqual(S[1][1], "pv:rw")

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
    getholdoff=None
    maxDiff = 4096

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

        self.dsconfig = self.setUpGW(self._us_conf)
        _log.debug("DS server conf: %s", self.dsconfig)

        # downstream client
        self._ds_client = Context('pva', self.dsconfig, useenv=False)

        _log.debug("Exit setUp")

    def setUpGW(self, usconfig):
        cfile = self._cfile = NamedTemporaryFile(mode='w+')
        json.dump({
            'version':2,
            'clients':[{
                'name':'client1',
                'provider':'pva',
                'addrlist':'127.0.0.1',
                'autoaddrlist':False,
                'bcastport':usconfig['EPICS_PVA_BROADCAST_PORT'],
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
                'getholdoff':self.getholdoff,
            }],
        }, cfile)
        cfile.flush()
        with open(cfile.name, 'r') as F:
            _log.debug('GW config')
            _log.debug(F.read())

        # gateway
        args = getargs().parse_args(['-v', cfile.name])

        self._app = TestApp(args)
        self._main = threading.Thread(target=self._app.run, name='GW Main')
        self._main.start()
        return self._app.servers[u'server1_0'].conf()

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

        self.tearDownGW()

        # upstream server
        self.stopServer()
        del self._us_provider
        del self._us_server
        del self.pv
        _defaultWorkQueue.sync()

        super(TestHighLevel, self).tearDown()
        _log.debug("Exit tearDown")

    def tearDownGW(self):
        # gateway
        self._app.abort()
        self._main.join(self.timeout)
        del self._app
        del self._main

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
        with self.assertRaisesRegex(RemoteError, ".*select.*"):
            val = self._ds_client.get('pv:name', timeout=self.timeout, request='nonexistant')

    def test_put(self):
        self._ds_client.put('pv:name', 41, timeout=self.timeout)

        val = self._ds_client.get('pv:name', timeout=self.timeout)
        self.assertEqual(val, 41)

    def test_put_bad_mask(self):
        with self.assertRaisesRegex(RemoteError, ".*select.*"):
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

class TestHighLevelChained(TestHighLevel):

    def setUpGW(self, usconfig):
        # First GW, connected to upstream server
        cfile = self._cfile = NamedTemporaryFile(mode='w+')
        json.dump({
            'version':2,
            'clients':[{
                'name':'client1',
                'provider':'pva',
                'addrlist':'127.0.0.1',
                'autoaddrlist':False,
                'bcastport':usconfig['EPICS_PVA_BROADCAST_PORT'],
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
                'getholdoff':self.getholdoff,
            }],
        }, cfile)
        cfile.flush()
        with open(cfile.name, 'r') as F:
            _log.debug('GW config')
            _log.debug(F.read())

        args = getargs().parse_args(['-v', cfile.name])

        self._app1 = TestApp(args)
        self._main1 = threading.Thread(target=self._app1.run, name='GW1 Main')
        self._main1.start()

        gw1config = self._app1.servers[u'server1_0'].conf()

        # Second GW, connected to first
        cfile = self._cfile = NamedTemporaryFile(mode='w+')
        json.dump({
            'version':2,
            'clients':[{
                'name':'client1',
                'provider':'pva',
                'addrlist':'127.0.0.1',
                'autoaddrlist':False,
                'bcastport':gw1config['EPICS_PVA_BROADCAST_PORT'],
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

        args = getargs().parse_args(['-v', cfile.name])

        self._app2 = TestApp(args)
        self._main2 = threading.Thread(target=self._app2.run, name='GW2 Main')
        self._main2.start()

        return self._app2.servers[u'server1_0'].conf()

    def tearDownGW(self):
        self._app2.abort()
        self._app1.abort()
        self._main2.join(self.timeout)
        self._main1.join(self.timeout)
        del self._app2
        del self._app1
        del self._main2
        del self._main1

class TestHighLevelGetHoldOff(TestHighLevel):
    getholdoff = 1 # hopefully long enough for CI without exceeding my patience

    def test_get_holdoff(self):
        N = 1 # shadow odometer counter
        _gw.addOdometer(self._us_server._S, 'odometer', 0)

        # first op will complete immeidately and start holdoff timer
        T0 = monotonic()
        Vs = self._ds_client.get('odometer', timeout=self.timeout)
        self.assertEqual(Vs, N)
        N = Vs + 1

        for _i in range(4):
            # issue a batch.  Due to runner timing, holdoff may already have expired :(
            # so make allowances...
            T1 = monotonic()
            Vs = self._ds_client.get(['odometer', 'odometer', 'odometer'], timeout=self.timeout)
            T2 = monotonic()
            Vmin = min(*Vs)
            Vmax = max(*Vs)

            self.assertTrue(Vmin==N and
                            Vmax<=N+1 and
                            T2-T1 >= self.getholdoff and
                            T2-T0 >= (Vmax - Vmin + 1) * self.getholdoff,
                            "%.3f, %.3f : %d %s"%(T1-T0, T2-T0, N, Vs))

            N = Vmax+1

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
        F = NamedTemporaryFile(mode='w+')
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
