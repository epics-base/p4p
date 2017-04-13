from __future__ import print_function

import unittest, random, weakref, gc, threading

from ..wrapper import Value, Type
from ..client.thread import Context
from ..server import Server, installProvider, removeProvider
from ..rpc import NTURIDispatcher, WorkQueue, rpc
from ..nt import NTScalar

class TestService(object):
   @rpc(NTScalar.buildType('d'))
   def add(self, lhs, rhs):
        return {
            'value': float(lhs)+float(rhs),
        }


class TestRPC(unittest.TestCase):
    def setUp(self):
        # random PV prefix
        # TODO: network isolation
        self.prefix = 'rpctest:%u:'%random.randint(0, 1024)

        service = TestService()

        # RPC handling queue
        self._Q = WorkQueue(maxsize=2)

        # RPC dispatcher (extract RPC args from PVD blob)
        dispatch = NTURIDispatcher(self._Q, target=service, prefix=self.prefix)
        self._dispatch = weakref.ref(dispatch)
        installProvider("TestRPC", dispatch)

        self.server = Server(providers="TestRPC")
        self.server.start()

        self._QT = threading.Thread(name="TestRPC Q", target=self._Q.handle)
        self._QT.start()

    def tearDown(self):
        self.server.stop()

        self._Q.interrupt()
        self._QT.join()

        removeProvider("TestRPC")
        self.assertIsNone(self._dispatch())

    def testAdd(self):
        args = Value(Type([
            ('schema', 's'),
            ('path', 's'),
            ('query', ('s', None, [
                ('lhs', 'd'),
                ('rhs', 'd'),
            ])),
        ]), {
            'schema': 'pva',
            'path': self.prefix+'add',
            'query': {
                'lhs': 1,
                'rhs': 1,
            },
        })
        ctxt = Context('pva')
        sum = ctxt.rpc(self.prefix+'add', args)
        self.assertEqual(sum.value, 2.0)

    def testAdd3(self):
        args = Value(Type([
            ('schema', 's'),
            ('path', 's'),
            ('query', ('s', None, [
                ('lhs', 'd'),
                ('rhs', 'd'),
            ])),
        ]), {
            'schema': 'pva',
            'path': self.prefix+'add',
            'query': {
                'lhs': 1,
                'rhs': 2,
            },
        })
        ctxt = Context('pva')
        sum = ctxt.rpc(self.prefix+'add', args)
        self.assertEqual(sum.value, 3.0)
