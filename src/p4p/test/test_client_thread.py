
from __future__ import print_function

import unittest
import weakref
import gc

from ..client.thread import Context, TimeoutError
from .utils import gctrace
from .utils import RefTestCase


class TestTimeout(RefTestCase):

    def setUp(self):
        super(TestTimeout, self).setUp()
        self.ctxt = Context('pva')

    def tearDown(self):
        self.ctxt.close()
        W = weakref.ref(self.ctxt)
        del self.ctxt
        gc.collect()
        C = W()
        if C is not None:
            print('trace', C)
            gctrace(C)
        self.assertIsNone(C)
        super(TestTimeout, self).tearDown()

    def test_get(self):
        R = self.ctxt.get('invalid:pv:name', timeout=0.1, throw=False)
        self.assertIsInstance(R, TimeoutError)

    def test_get_throw(self):
        self.assertRaises(TimeoutError, self.ctxt.get, 'invalid:pv:name', timeout=0.1)

    def test_put(self):
        R = self.ctxt.put('invalid:pv:name', 0, timeout=0.1, throw=False)
        self.assertIsInstance(R, TimeoutError)

    def test_put_throw(self):
        self.assertRaises(TimeoutError, self.ctxt.put, 'invalid:pv:name', 0, timeout=0.1)
