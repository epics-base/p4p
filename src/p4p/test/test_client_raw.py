
from __future__ import print_function

import unittest
import weakref
import gc

from ..client.raw import Context, Cancelled
from ..wrapper import Value, Type
from .utils import RefTestCase


class TestRequest(RefTestCase):

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


class TestProviders(RefTestCase):

    def tearDown(self):
        gc.collect()  # try to provoke any crashes here so they can be associated with this testcase

    def testProviders(self):
        providers = Context.providers()
        self.assertIn('pva', providers)


class TestPVA(RefTestCase):

    def setUp(self):
        super(TestPVA, self).setUp()
        self.ctxt = Context("pva")

    def tearDown(self):
        self.ctxt.close()
        self.ctxt = None
        gc.collect()
        super(TestPVA, self).tearDown()

    def testGetAbort(self):
        _X = [None]

        def fn(V):
            _X[0] = V
        op = self.ctxt.get("completelyInvalidChannelName", fn)

        op.close()

        self.assertIsInstance(_X[0], Cancelled)

    def testGetAbortGC(self):
        _X = [None]

        def fn(V):
            _X[0] = V
        op = self.ctxt.get("completelyInvalidChannelName", fn)

        W = weakref.ref(op)
        del op
        gc.collect()

        self.assertIsNone(W())

        self.assertIsNone(_X[0])

    def testGCCycle(self):
        _X = [None]

        def fn(V):
            _X[0] = V
        op = self.ctxt.get("completelyInvalidChannelName", fn)

        fn._cycle = op  # create cycle: op -> fn -> fn.__dict__ -> op

        self.assertIn(fn.__dict__, gc.get_referrers(op))

        W = weakref.ref(op), weakref.ref(fn)
        del op, fn
        gc.collect()

        self.assertIsNone(W[0]())
        self.assertIsNone(W[1]())

        self.assertIsNone(_X[0])

    def testRPCAbort(self):
        P = Value(Type([
            ('value', 'i'),
        ]), {
            'value': 42,
        })

        _X = [None]

        def fn(V):
            _X[0] = V
        op = self.ctxt.rpc("completelyInvalidChannelName", fn, P)

        W = weakref.ref(op)
        del op
        gc.collect()

        self.assertIsNone(W())

        self.assertIsNone(_X[0])

    def testMonAbort(self):
        canery = object()
        _X = [canery]

        def evt(V):
            _X[0] = V

        op = self.ctxt.monitor("completelyInvalidChannelName", evt)

        op.close()

        self.assertIsInstance(_X[0], Cancelled)

    def testMonCycle(self):
        canery = object()
        _X = [canery]

        def evt(V):
            _X[0] = V

        op = self.ctxt.monitor("completelyInvalidChannelName", evt)

        evt._cycle = op  # op -> evt -> evt.__dict__ -> op

        self.assertIn(evt.__dict__, gc.get_referrers(op))

        W = weakref.ref(op)
        del op, evt
        gc.collect()

        self.assertIsNone(W())

        self.assertIs(_X[0], canery)
