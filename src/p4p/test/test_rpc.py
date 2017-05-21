from __future__ import print_function

import unittest, random, weakref, gc, threading

from ..wrapper import Value, Type
from ..client.thread import Context
from ..server import Server, installProvider, removeProvider
from ..rpc import NTURIDispatcher, WorkQueue, rpc, rpccall, rpcproxy
from ..nt import NTScalar, NTURI

class TestService(object):
   @rpc(NTScalar('d'))
   def add(self, lhs, rhs):
        return float(lhs)+float(rhs)


class TestRPC(unittest.TestCase):
    def setUp(self):
        # TODO: need PVA API change before we can run w/ network isolation
        conf = {
            'EPICS_PVAS_INTF_ADDR_LIST':'127.0.0.1',
            'EPICS_PVA_ADDR_LIST':'127.0.0.1',
            'EPICS_PVA_AUTO_ADDR_LIST':'0',
            'EPICS_PVA_SERVER_PORT':'0',
            'EPICS_PVA_BROADCAST_PORT':'0',
        }

        # random PV prefix
        self.prefix = 'rpctest:%u:'%random.randint(0, 1024)

        service = TestService()

        # RPC handling queue
        self._Q = WorkQueue(maxsize=2)

        # RPC dispatcher (extract RPC args from PVD blob)
        dispatch = NTURIDispatcher(self._Q, target=service, prefix=self.prefix)
        self._dispatch = weakref.ref(dispatch)
        installProvider("TestRPC", dispatch)

        self.server = Server(providers="TestRPC", conf=conf, useenv=False)
        print("conf", self.server.conf(client=True, server=False))
        self.server.start()

        self._QT = threading.Thread(name="TestRPC Q", target=self._Q.handle)
        self._QT.start()

    def tearDown(self):
        self.server.stop()

        self._Q.interrupt()
        self._QT.join()

        removeProvider("TestRPC")
        gc.collect()
        self.assertIsNone(self._dispatch())

    def testAdd(self):
        args = NTURI([
            ('lhs', 'd'),
            ('rhs', 'd'),
        ]).wrap(self.prefix+'add', {
            'lhs': 1,
            'rhs': 1,
        }, scheme='pva')
        ctxt = Context('pva', useenv=False, conf=self.server.conf(client=True, server=False), unwrap=False)
        self.assertEqual(ctxt.name, 'pva')
        sum = ctxt.rpc(self.prefix+'add', args)
        self.assertEqual(sum.value, 2.0)

    def testAdd3(self):
        args = NTURI([
            ('lhs', 'd'),
            ('rhs', 'd'),
        ]).wrap(self.prefix+'add', {
            'lhs': 1,
            'rhs': 2,
        }, scheme='pva')
        ctxt = Context('pva', useenv=False, conf=self.server.conf(client=True, server=False), unwrap=False)
        sum = ctxt.rpc(self.prefix+'add', args)
        self.assertEqual(sum.value, 3.0)

class TestProxy(unittest.TestCase):
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
            ('query', [('A', 4), ('B', u'')])
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
