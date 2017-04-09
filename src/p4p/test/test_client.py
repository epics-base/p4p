
from __future__ import print_function

import unittest
import weakref, gc

from ..client import Context

class TestProviders(unittest.TestCase):
    def tearDown(self):
        gc.collect() # try to provoke any crashes here so they can be associated with this testcase
    def testProviders(self):
        providers = Context.providers()
        self.assertIn('pva', providers)
        self.assertIn('ca', providers)

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

        self.assertTrue(op.cancel())

        self.assertIsNone(_X[0])

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
