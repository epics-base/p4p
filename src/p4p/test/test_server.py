from __future__ import print_function

import unittest, random, weakref, gc, threading

from ..server import Server, installProvider, removeProvider
from ..client.thread import Context

class TestDummyProvider(unittest.TestCase):
    # will fail if anything is done to it.
    class Dummy(object):
        pass

    def test_install(self):
        D = self.Dummy()
        d = weakref.ref(D)
        installProvider("foo", D)
        del D
        removeProvider("foo")
        gc.collect()
        self.assertIsNone(d())

    def test_client(self):
        D = self.Dummy()
        d = weakref.ref(D)
        installProvider("foo", D)
        del D

        try:
            with Context('foo'):

                removeProvider("foo")
                gc.collect()
                self.assertIsNotNone(d())

            gc.collect()
            self.assertIsNone(d())
        finally:
            removeProvider("foo")
