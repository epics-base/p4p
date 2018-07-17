from __future__ import print_function

import unittest
import random
import weakref
import sys
import gc
import threading

from ..wrapper import Value, Type
from ..client.thread import Context
from ..server import Server, installProvider, removeProvider
from ..rpc import NTURIDispatcher, WorkQueue, rpc, rpccall, rpcproxy
from ..nt import NTScalar, NTURI
from .utils import RefTestCase


class TestService(object):

    @rpc(NTScalar('d'))
    def add(self, lhs, rhs):
        return float(lhs) + float(rhs)


class TestRPCFull(RefTestCase):

    """Test end to end

    full server and client communicating through the loopback
    """
    runserver = True
    provider = 'pva'
    getconfig = lambda self: self.server.conf()

    def setUp(self):
        super(TestRPCFull, self).setUp()

        conf = {
            'EPICS_PVAS_INTF_ADDR_LIST': '127.0.0.1',
            'EPICS_PVA_ADDR_LIST': '127.0.0.1',
            'EPICS_PVA_AUTO_ADDR_LIST': '0',
            'EPICS_PVA_SERVER_PORT': '0',
            'EPICS_PVA_BROADCAST_PORT': '0',
        }

        # random PV prefix
        self.prefix = 'rpctest:%u:' % random.randint(0, 1024)

        service = TestService()

        # RPC handling queue
        self._Q = WorkQueue(maxsize=2)

        # RPC dispatcher (extract RPC args from PVD blob)
        dispatch = NTURIDispatcher(self._Q, target=service, prefix=self.prefix, name="TestRPC")
        self._dispatch = weakref.ref(dispatch)

        if self.runserver:
            self.server = Server(providers=[dispatch], conf=conf, useenv=False)
            print("conf", self.server.conf())
        else:
            installProvider("TestRPC", dispatch)

        self._QT = threading.Thread(name="TestRPC Q", target=self._Q.handle)
        self._QT.start()

    def tearDown(self):
        if self.runserver:
            self.server.stop()
            self.server = None

        self._Q.interrupt()
        self._QT.join()

        if not self.runserver:
            removeProvider("TestRPC")
        gc.collect()
        D = self._dispatch()
        if D is not None:
            print("dispatcher lives! ", sys.getrefcount(D), " refs  referrers:")
            import inspect
            for R in gc.get_referrers(D):
                print(R)
        self.assertIsNone(D)
        super(TestRPCFull, self).tearDown()

    def testAdd(self):
        args = NTURI([
            ('lhs', 'd'),
            ('rhs', 'd'),
        ]).wrap(self.prefix + 'add', kws={
            'lhs': 1,
            'rhs': 1,
        }, scheme='pva')
        with Context(self.provider, useenv=False, conf=self.getconfig(), unwrap=False) as ctxt:
            self.assertEqual(ctxt.name, self.provider)
            sum = ctxt.rpc(self.prefix + 'add', args)
            self.assertEqual(sum.value, 2.0)

    def testAdd3(self):
        args = NTURI([
            ('lhs', 'd'),
            ('rhs', 'd'),
        ]).wrap(self.prefix + 'add', kws={
            'lhs': 1,
            'rhs': 2,
        }, scheme='pva')
        with Context(self.provider, useenv=False, conf=self.getconfig(), unwrap=False) as ctxt:
            sum = ctxt.rpc(self.prefix + 'add', args)
            self.assertEqual(sum.value, 3.0)


class TestRPCProvider(TestRPCFull):

    """end to end w/o network
    """
    runserver = False
    provider = 'server:TestRPC'
    getconfig = lambda self: {}


class TestProxy(RefTestCase):

    class MockContext(object):
        name = 'fake'

        def rpc(self, *args, **kws):
            return args, kws

    @rpcproxy
    class MyProxy(object):

        def __init__(self, myarg=5):
            self.myarg = myarg

        @rpccall('%sfoo')
        def bar(A='i', B='s'):
            pass

        @rpccall('%sbaz')
        def another(X='s', Y='i'):
            pass

    def setUp(self):
        super(TestProxy, self).setUp()
        ctxt = self.MockContext()
        self.proxy = self.MyProxy(myarg=3, context=ctxt, format='pv:')
        self.assertEqual(self.proxy.myarg, 3)
        self.assertIs(self.proxy.context, ctxt)

    def test_call1(self):
        args, kws = self.proxy.bar(4, 'one')

        print(args, kws)
        self.assertEqual(args[0], 'pv:foo')
        self.assertListEqual(args[1].tolist(), [
            ('scheme', u'fake'),
            ('authority', u''),
            ('path', u'pv:foo'),
            ('query', [('A', 4), ('B', u'one')])
        ])
        self.assertEqual(len(args), 2)
        self.assertDictEqual(kws, {'request': None, 'throw': True, 'timeout': 3.0})

    def test_call2(self):
        args, kws = self.proxy.bar(4, B='one')

        print(args, kws)
        self.assertEqual(args[0], 'pv:foo')
        self.assertListEqual(args[1].tolist(), [
            ('scheme', u'fake'),
            ('authority', u''),
            ('path', u'pv:foo'),
            ('query', [('A', 4), ('B', u'one')])
        ])
        self.assertEqual(len(args), 2)
        self.assertDictEqual(kws, {'request': None, 'throw': True, 'timeout': 3.0})

    def test_call3(self):
        args, kws = self.proxy.bar(4)

        print(args, kws)
        self.assertEqual(args[0], 'pv:foo')
        self.assertListEqual(args[1].tolist(), [
            ('scheme', u'fake'),
            ('authority', u''),
            ('path', u'pv:foo'),
            ('query', [('A', 4)])
        ])
        self.assertEqual(len(args), 2)
        self.assertDictEqual(kws, {'request': None, 'throw': True, 'timeout': 3.0})

    def test_call4(self):
        args, kws = self.proxy.another('one', Y=2)

        print(args, kws)
        self.assertEqual(args[0], 'pv:baz')
        self.assertListEqual(args[1].tolist(), [
            ('scheme', u'fake'),
            ('authority', u''),
            ('path', u'pv:baz'),
            ('query', [('X', u'one'), ('Y', 2)])
        ])
        self.assertEqual(len(args), 2)
        self.assertDictEqual(kws, {'request': None, 'throw': True, 'timeout': 3.0})
