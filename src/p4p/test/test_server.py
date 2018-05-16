from __future__ import print_function

import unittest, sys, random, weakref, gc, threading

from ..server import Server, installProvider, removeProvider
from ..client.thread import Context

def checkweak(O):
    o = O()
    if o is not None:
        print('Live object', id(o), type(o), sys.getrefcount(o), gc.get_referrers(o))
    return o

class TestDummyProvider(unittest.TestCase):
    # will fail if anything is done to it.
    class Dummy(object):
        pass

    def test_install(self):
        "Install and remove provider"
        D = self.Dummy()
        d = weakref.ref(D)
        installProvider("foo", D)
        del D
        removeProvider("foo")
        gc.collect()
        self.assertIsNone(d())

    def test_server(self):
        D = self.Dummy()
        d = weakref.ref(D)
        installProvider("foo", D)
        try:
            with Server(providers=["foo"]) as S:
                s = weakref.ref(S)
        finally:
            removeProvider("foo")
        del D
        del S
        gc.collect()
        self.assertIsNone(checkweak(d))
        self.assertIsNone(checkweak(s))
        

    def test_client(self):
        D = self.Dummy()
        d = weakref.ref(D)
        installProvider("foo", D)
        del D

        try:
            with Context('server:foo'):

                removeProvider("foo")
                gc.collect()
                self.assertIsNotNone(d())

            gc.collect()
            self.assertIsNone(d())
        finally:
            removeProvider("foo")
