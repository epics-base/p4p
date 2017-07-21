
from __future__ import print_function

import unittest
import weakref, gc

from ..client.raw import Context
from ..wrapper import Value, Type
from .. import Cancelled

class TestRequest(unittest.TestCase):
    def testEmpty(self):
        self.assertListEqual(Context.makeRequest("").tolist(), [])

    def testValue(self):
        self.assertListEqual(Context.makeRequest("field(value)").tolist(),
            [('field', [('value', [])])]
        )

    def testAll(self):
        self.assertListEqual(Context.makeRequest("field()").tolist(),
            [('field', [])]
        )

class TestProviders(unittest.TestCase):
    def tearDown(self):
        gc.collect() # try to provoke any crashes here so they can be associated with this testcase
    def testProviders(self):
        providers = Context.providers()
        self.assertIn('pva', providers)

class TestPVA(unittest.TestCase):
    def setUp(self):
        self.ctxt = Context("pva")
    def tearDown(self):
        self.ctxt = None
        gc.collect()

    def testChan(self):
        chan = self.ctxt.channel("completelyInvalidChannelName")

        self.assertEqual(chan.getName(), "completelyInvalidChannelName")

    def testGetAbort(self):
        chan = self.ctxt.channel("completelyInvalidChannelName")
        _X = [None]
        def fn(V):
            _X[0] = V
        op = chan.get(fn)

        op.cancel()

        self.assertIsInstance(_X[0], Cancelled)

    def testGetAbortGC(self):
        chan = self.ctxt.channel("completelyInvalidChannelName")
        _X = [None]
        def fn(V):
            _X[0] = V
        op = chan.get(fn)

        W =  weakref.ref(op)
        del op
        gc.collect()

        self.assertIsNone(W())

        self.assertIsNone(_X[0])

    def testGCCycle(self):
        chan = self.ctxt.channel("completelyInvalidChannelName")
        _X = [None]
        def fn(V):
            _X[0] = V
        op = chan.get(fn)

        fn._cycle = op # create cycle: op -> fn -> fn.__dict__ -> op

        self.assertIn(fn.__dict__, gc.get_referrers(op))

        W =  weakref.ref(op)
        del op, fn
        gc.collect()

        self.assertIsNone(W())

        self.assertIsNone(_X[0])

    def testRPCAbort(self):
        P = Value(Type([
            ('value', 'i'),
        ]), {
            'value': 42,
        })
        chan = self.ctxt.channel("completelyInvalidChannelName")
        _X = [None]
        def fn(V):
            _X[0] = V
        op = chan.rpc(fn, P)

        W =  weakref.ref(op)
        del op
        gc.collect()

        self.assertIsNone(W())

        self.assertIsNone(_X[0])

    def testMonAbort(self):
        chan = self.ctxt.channel("completelyInvalidChannelName")

        canery = object()
        _X = [canery]
        def evt(V):
            _X[0] = V

        op = chan.monitor(evt)

        op.close()

        self.assertIs(_X[0], canery)

    def testMonCycle(self):
        chan = self.ctxt.channel("completelyInvalidChannelName")

        canery = object()
        _X = [canery]
        def evt(V):
            _X[0] = V

        op = chan.monitor(evt)

        evt._cycle = op # op -> evt -> evt.__dict__ -> op

        self.assertIn(evt.__dict__, gc.get_referrers(op))

        W =  weakref.ref(op)
        del op, evt
        gc.collect()

        self.assertIsNone(W())

        self.assertIs(_X[0], canery)
