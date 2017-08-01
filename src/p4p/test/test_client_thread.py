
from __future__ import print_function

import unittest
import weakref, gc

from ..client.thread import Context, TimeoutError

class TestTimeout(unittest.TestCase):
    def setUp(self):
        self.ctxt = Context('pva')

    def tearDown(self):
        W = weakref.ref(self.ctxt)
        del self.ctxt
        gc.collect()
        self.assertIsNone(W())

    def test_get(self):
        self.assertRaises(TimeoutError, self.ctxt.get, 'invalid:pv:name', timeout=0.1)

    def test_put(self):
        self.assertRaises(TimeoutError, self.ctxt.put, 'invalid:pv:name', 0, timeout=0.1)
